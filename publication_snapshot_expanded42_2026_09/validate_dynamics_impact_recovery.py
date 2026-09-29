from pathlib import Path
import json
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
    / "highrate_semantic_event_manifest_v2.csv"
)

OUT = (
    CAMPAIGN
    / "dynamics_impact_recovery_audit.csv"
)


def norm_path(x):
    p = Path(str(x))
    if not p.is_absolute():
        p = ROOT / p
    return p


def finite_float(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def recursive_first(obj, keys):
    keys = {
        str(k).lower()
        for k in keys
    }

    found = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if str(k).lower() in keys:
                    vv = finite_float(v)
                    if vv is not None:
                        found.append(vv)
                walk(v)

        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)

    return found[0] if found else None


def read_json(path):
    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )
    except Exception:
        return None


def recover_existing_onset(current_truth):
    # --------------------------------------------------------
    # 1. High-rate labeling metadata
    # --------------------------------------------------------
    meta = current_truth.with_name(
        current_truth.stem
        + ".label_metadata.json"
    )

    if meta.exists():
        obj = read_json(meta)

        if obj is not None:
            x = recursive_first(
                obj,
                {
                    "fall_onset_time_s",
                },
            )

            if x is not None:
                return (
                    x,
                    "highrate_label_metadata",
                )

    # --------------------------------------------------------
    # 2. Run manifest
    # --------------------------------------------------------
    manifest = (
        current_truth.parent
        / "run_manifest.json"
    )

    if manifest.exists():
        obj = read_json(manifest)

        if obj is not None:
            x = recursive_first(
                obj,
                {
                    "fall_onset_time_s",
                },
            )

            if x is not None:
                return (
                    x,
                    "run_manifest_imu_labeling",
                )

    return None, "unresolved"


def dynamics_path_from_truth(current_truth):
    name = current_truth.name

    suffix = "_highrate_truth.csv"

    if not name.endswith(suffix):
        return None

    stem = name[
        :-len(suffix)
    ]

    return current_truth.with_name(
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
            return lower[c.lower()]

    return None


def recover_impact_from_dynamics(
    dynamics_csv,
    onset,
    subject_weight,
):
    """
    Replicate the common simulator event detector's
    main-impact logic:

      after onset:
        first primary body-impact load > 0.25 body weight

    The detector estimates BW from early support load when
    possible.  If that information is unavailable, use the
    known simulation-profile body mass.
    """

    df = pd.read_csv(
        dynamics_csv
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
            "primary impact body-load column missing"
        )

    t = pd.to_numeric(
        df[time_col],
        errors="coerce",
    ).to_numpy(float)

    force = pd.to_numeric(
        df[force_col],
        errors="coerce",
    ).fillna(0).to_numpy(float)

    # --------------------------------------------------------
    # Match runtime detector's body-mass estimate:
    # median positive support load among first <=30 frames.
    # --------------------------------------------------------
    mass_source = "profile_weight"

    body_mass = (
        float(subject_weight)
        if finite_float(subject_weight) is not None
        else 70.0
    )

    if support_col is not None:
        support = pd.to_numeric(
            df[support_col],
            errors="coerce",
        ).fillna(0).to_numpy(float)

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
            body_mass = max(
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
        body_mass * 9.81,
    )

    imp_bw = force / BW

    valid_t = np.isfinite(t)

    if not np.any(valid_t):
        raise RuntimeError(
            "no finite dynamics timestamps"
        )

    j0 = int(
        np.searchsorted(
            t,
            float(onset),
            side="left",
        )
    )

    if j0 >= len(t):
        raise RuntimeError(
            "onset occurs after dynamics timeline"
        )

    crossing = np.where(
        imp_bw[j0:] > 0.25
    )[0]

    if len(crossing):
        j = (
            j0
            + int(crossing[0])
        )

        method = (
            "first_primary_body_"
            "impact_gt_0p25BW"
        )

    else:
        # This is the same fallback used in the common
        # detector. Keep it explicit in the audit.
        segment = imp_bw[j0:]

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
        "impact_time_s": float(
            t[j]
        ),
        "impact_index": int(j),
        "impact_force_n": float(
            force[j]
        ),
        "impact_bw": float(
            imp_bw[j]
        ),
        "estimated_body_mass_kg": float(
            body_mass
        ),
        "mass_source": mass_source,
        "method": method,
        "time_column": time_col,
        "force_column": force_col,
    }


if not MANIFEST.exists():
    raise SystemExit(
        f"Missing manifest:\n{MANIFEST}"
    )


m = pd.read_csv(
    MANIFEST
)

rows = []


for _, r in m.iterrows():

    current = norm_path(
        r["current_truth_csv"]
    )

    known_onset = finite_float(
        r.get(
            "fall_onset_time_s"
        )
    )

    known_impact = finite_float(
        r.get(
            "main_impact_time_s"
        )
    )

    # --------------------------------------------------------
    # For the unresolved early records, recover the onset
    # already produced by the labeling system.
    # --------------------------------------------------------
    if known_onset is not None:
        onset = known_onset
        onset_source = (
            str(
                r.get(
                    "event_source",
                    "existing_manifest",
                )
            )
        )
    else:
        onset, onset_source = (
            recover_existing_onset(
                current
            )
        )

    dyn = dynamics_path_from_truth(
        current
    )

    result = {
        "scenario_id": int(
            r["scenario_id"]
        ),
        "sim_age_profile": int(
            r["sim_age_profile"]
        ),
        "weight_kg": r.get(
            "weight_kg",
            np.nan,
        ),
        "known_onset_time_s": known_onset,
        "onset_used_s": onset,
        "onset_source": onset_source,
        "known_impact_time_s": known_impact,
        "dynamics_csv": (
            str(dyn)
            if dyn is not None
            else ""
        ),
        "candidate_impact_time_s": np.nan,
        "impact_error_s": np.nan,
        "impact_abs_error_s": np.nan,
        "impact_force_n": np.nan,
        "impact_bw": np.nan,
        "estimated_body_mass_kg": np.nan,
        "mass_source": "",
        "impact_method": "",
        "status": "FAIL",
        "error": "",
    }

    try:
        if onset is None:
            raise RuntimeError(
                "no usable onset time"
            )

        if (
            dyn is None
            or not dyn.exists()
        ):
            raise RuntimeError(
                "dynamics frame CSV missing"
            )

        rec = (
            recover_impact_from_dynamics(
                dyn,
                onset,
                r.get(
                    "weight_kg",
                    np.nan,
                ),
            )
        )

        candidate = rec[
            "impact_time_s"
        ]

        result[
            "candidate_impact_time_s"
        ] = candidate

        result[
            "impact_force_n"
        ] = rec[
            "impact_force_n"
        ]

        result[
            "impact_bw"
        ] = rec[
            "impact_bw"
        ]

        result[
            "estimated_body_mass_kg"
        ] = rec[
            "estimated_body_mass_kg"
        ]

        result[
            "mass_source"
        ] = rec[
            "mass_source"
        ]

        result[
            "impact_method"
        ] = rec[
            "method"
        ]

        if known_impact is not None:
            err = (
                candidate
                - known_impact
            )

            result[
                "impact_error_s"
            ] = err

            result[
                "impact_abs_error_s"
            ] = abs(err)

        result[
            "status"
        ] = "OK"

    except Exception as exc:
        result[
            "error"
        ] = str(exc)

    rows.append(
        result
    )


out = pd.DataFrame(
    rows
)

out.to_csv(
    OUT,
    index=False,
)


known = out[
    out[
        "known_impact_time_s"
    ].notna()
    &
    (
        out["status"]
        == "OK"
    )
].copy()

missing = out[
    out[
        "known_impact_time_s"
    ].isna()
].copy()


print("=" * 100)
print("DYNAMIC IMPACT RECOVERY VALIDATION")
print("=" * 100)

print(
    "records total           :",
    len(out),
)

print(
    "candidate recovery OK   :",
    int(
        (
            out.status
            == "OK"
        ).sum()
    ),
)

print(
    "candidate recovery FAIL :",
    int(
        (
            out.status
            != "OK"
        ).sum()
    ),
)

print()
print(
    "known impacts available :",
    len(known),
)

print(
    "missing impacts to fill :",
    len(missing),
)

print()
print("Impact methods:")
print(
    out[
        "impact_method"
    ]
    .value_counts()
    .to_string()
)

print()
print("Onset sources for missing-impact records:")
print(
    missing[
        "onset_source"
    ]
    .value_counts()
    .to_string()
)

print()
print("=" * 100)
print("221-RECORD GROUND-TRUTH CROSS-CHECK")
print("=" * 100)

if len(known):

    print(
        "median abs error [s] :",
        known[
            "impact_abs_error_s"
        ].median(),
    )

    print(
        "mean abs error [s]   :",
        known[
            "impact_abs_error_s"
        ].mean(),
    )

    print(
        "p95 abs error [s]    :",
        known[
            "impact_abs_error_s"
        ].quantile(
            0.95
        ),
    )

    print(
        "max abs error [s]    :",
        known[
            "impact_abs_error_s"
        ].max(),
    )

    for tol in [
        0.005,
        0.02,
        0.04,
        0.10,
    ]:
        frac = float(
            (
                known[
                    "impact_abs_error_s"
                ]
                <= tol
            ).mean()
        )

        print(
            f"within {tol:0.3f}s         : "
            f"{frac:.3%}"
        )


print()
print("Cross-check by scenario:")

if len(known):
    by_task = (
        known.groupby(
            "scenario_id"
        )[
            "impact_abs_error_s"
        ]
        .agg(
            [
                "count",
                "median",
                "max",
            ]
        )
    )

    print(
        by_task.to_string()
    )


print()
print("=" * 100)
print("117 EARLY-RECORD RECOVERY CANDIDATES")
print("=" * 100)

print(
    "with onset:",
    int(
        missing[
            "onset_used_s"
        ].notna().sum()
    ),
)

print(
    "with impact candidate:",
    int(
        missing[
            "candidate_impact_time_s"
        ].notna().sum()
    ),
)

print()
print("Candidate impact delay onset->impact by task:")

tmp = missing.copy()

tmp[
    "onset_to_impact_s"
] = (
    tmp[
        "candidate_impact_time_s"
    ]
    -
    tmp[
        "onset_used_s"
    ]
)

print(
    tmp.groupby(
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


bad = out[
    out.status != "OK"
]

if len(bad):
    print(
        "\n===== RECOVERY FAILURES ====="
    )

    print(
        bad[
            [
                "scenario_id",
                "sim_age_profile",
                "dynamics_csv",
                "error",
            ]
        ].to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# Acceptance criterion.
#
# Dynamics are sampled at the 30-Hz control rate, so one-frame
# quantization is ~0.0333 s. Require:
#   - all 338 candidates recoverable
#   - all 221 known records cross-checked
#   - median <= 5 ms
#   - 95th percentile <= one control frame + small rounding
#   - max <= 50 ms
# ------------------------------------------------------------
validation_ok = (
    len(out) == 338
    and bool(
        (
            out.status
            == "OK"
        ).all()
    )
    and len(known) == 221
    and float(
        known[
            "impact_abs_error_s"
        ].median()
    ) <= 0.005
    and float(
        known[
            "impact_abs_error_s"
        ].quantile(
            0.95
        )
    ) <= 0.040
    and float(
        known[
            "impact_abs_error_s"
        ].max()
    ) <= 0.050
    and int(
        missing[
            "candidate_impact_time_s"
        ].notna().sum()
    ) == 117
)


print()
print(
    "DYNAMICS_IMPACT_RULE_VALIDATED:",
    validation_ok,
)

if validation_ok:
    print(
        "The existing simulator's physics-based "
        "impact rule is validated for recovery."
    )
    print(
        "No simulator rerun is required."
    )
else:
    print(
        "Do not fill the 117 impacts yet; "
        "review the audit first."
    )

print()
print(
    "Audit CSV:",
    OUT,
)
