from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

import batch_run_all_labeled as batch


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
    / "common_onset_reconstruction_audit.csv"
)


def finite(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def norm_path(x):
    p = Path(str(x))

    if not p.is_absolute():
        p = ROOT / p

    return p


def derive_artifact(
    current_truth,
    suffix,
):
    name = current_truth.name

    ending = "_highrate_truth.csv"

    if not name.endswith(ending):
        return None

    stem = name[:-len(ending)]

    return current_truth.with_name(
        stem + suffix
    )


def metadata_onset(
    current_truth,
):
    """
    Recover an already-existing onset from label metadata.
    Used only for comparison / preserving Task 20 later.
    """

    p = current_truth.with_name(
        current_truth.stem
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


def perturb_from_log(
    log_path,
):
    if (
        log_path is None
        or not log_path.exists()
    ):
        return None, "missing_log"

    text = log_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    # parse_event_summary prefers the common summary;
    # for legacy tasks it falls back to phase parsing.
    try:
        ev = (
            batch.parse_event_summary(
                text
            )
            or {}
        )
    except Exception:
        ev = {}

    t = finite(
        ev.get(
            "perturbation_start_time_s"
        )
    )

    src = str(
        ev.get(
            "source",
            "unresolved",
        )
    )

    return t, src


def reconstruct_common_onset(
    markers_csv,
    segments_csv,
    perturb_t,
):
    """
    Reconstruct the common simulator biomechanical onset rule
    from saved MarkerKinematics + segment kinematics.

    Rule:
      pre_h = median pelvis height over previous <=30 frames

      h_threshold =
          max(
              0.78 * pre_h,
              pre_h - 0.10
          )

      h_cond:
          pelvis_height < h_threshold

      trunk_cond:
          trunk_lean_deg > 22 deg

      v_cond:
          horizontal pelvis speed > 0.55 m/s

      onset:
          first post-perturb frame satisfying

          (trunk_cond AND h_cond)
          OR
          (h_cond AND v_cond)

      if no frame satisfies:
          perturbation start is fallback.
    """

    if perturb_t is None:
        raise RuntimeError(
            "perturbation time unresolved"
        )

    mk = pd.read_csv(
        markers_csv
    )

    sg = pd.read_csv(
        segments_csv
    )

    required_mk = [
        "time",
        "pelvis_height",
        "trunk_lean_deg",
    ]

    required_sg = [
        "time",
        "pelvis_vx",
        "pelvis_vy",
    ]

    miss = [
        c for c in required_mk
        if c not in mk.columns
    ]

    if miss:
        raise RuntimeError(
            f"marker columns missing: {miss}"
        )

    miss = [
        c for c in required_sg
        if c not in sg.columns
    ]

    if miss:
        raise RuntimeError(
            f"segment columns missing: {miss}"
        )

    tm = pd.to_numeric(
        mk["time"],
        errors="coerce",
    ).to_numpy(float)

    ts = pd.to_numeric(
        sg["time"],
        errors="coerce",
    ).to_numpy(float)

    h = pd.to_numeric(
        mk["pelvis_height"],
        errors="coerce",
    ).to_numpy(float)

    trunk = pd.to_numeric(
        mk["trunk_lean_deg"],
        errors="coerce",
    ).to_numpy(float)

    vx = pd.to_numeric(
        sg["pelvis_vx"],
        errors="coerce",
    ).to_numpy(float)

    vy = pd.to_numeric(
        sg["pelvis_vy"],
        errors="coerce",
    ).to_numpy(float)

    if len(tm) != len(ts):
        raise RuntimeError(
            "marker/segment frame counts differ"
        )

    finite_time = (
        np.isfinite(tm)
        & np.isfinite(ts)
    )

    if not np.any(finite_time):
        raise RuntimeError(
            "no finite timeline"
        )

    max_time_delta = float(
        np.nanmax(
            np.abs(
                tm[finite_time]
                - ts[finite_time]
            )
        )
    )

    # They should represent the exact same native frames.
    if max_time_delta > 1e-4:
        raise RuntimeError(
            "marker/segment timelines do not align: "
            f"max delta={max_time_delta}"
        )

    t = tm

    speed = np.sqrt(
        vx ** 2
        + vy ** 2
    )

    # --------------------------------------------------------
    # Find first native frame at/after perturbation.
    # --------------------------------------------------------
    post_idx = np.where(
        np.isfinite(t)
        &
        (t >= perturb_t)
    )[0]

    if not len(post_idx):
        raise RuntimeError(
            "perturbation occurs after saved timeline"
        )

    j0 = int(
        post_idx[0]
    )

    # --------------------------------------------------------
    # Runtime detector uses up to the preceding 30 native
    # frames for baseline pelvis height.
    # --------------------------------------------------------
    a = max(
        0,
        j0 - 30,
    )

    pre_h = h[
        a:j0
    ]

    pre_h = pre_h[
        np.isfinite(pre_h)
    ]

    if not len(pre_h):
        raise RuntimeError(
            "no valid pre-perturbation pelvis-height baseline"
        )

    h_ref = float(
        np.median(
            pre_h
        )
    )

    threshold = float(
        max(
            0.78 * h_ref,
            h_ref - 0.10,
        )
    )

    h_cond = (
        np.isfinite(h)
        &
        (h < threshold)
    )

    trunk_cond = (
        np.isfinite(trunk)
        &
        (trunk > 22.0)
    )

    v_cond = (
        np.isfinite(speed)
        &
        (speed > 0.55)
    )

    combined = (
        h_cond
        &
        (
            trunk_cond
            |
            v_cond
        )
    )

    candidates = np.where(
        (t >= perturb_t)
        &
        combined
    )[0]

    if len(candidates):
        j = int(
            candidates[0]
        )

        method = (
            "common_rule_first_"
            "postperturb_condition"
        )

        onset = float(
            t[j]
        )

    else:
        # Existing detector fallback.
        j = j0
        onset = float(
            perturb_t
        )

        method = (
            "common_rule_fallback_"
            "perturbation_start"
        )

    return {
        "onset_s": onset,
        "index": j,
        "method": method,
        "h_ref_m": h_ref,
        "height_threshold_m":
            threshold,
        "pelvis_height_m":
            float(h[j]),
        "trunk_lean_deg":
            float(trunk[j]),
        "horizontal_speed_mps":
            float(speed[j]),
        "h_cond": bool(
            h_cond[j]
        ),
        "trunk_cond": bool(
            trunk_cond[j]
        ),
        "v_cond": bool(
            v_cond[j]
        ),
        "max_marker_segment_time_delta_s":
            max_time_delta,
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

    markers = derive_artifact(
        current,
        "_markers.csv",
    )

    segments = derive_artifact(
        current,
        "_segments.csv",
    )

    log_raw = r.get(
        "batch_log",
        "",
    )

    log = None

    if (
        isinstance(log_raw, str)
        and log_raw.strip()
    ):
        log = norm_path(
            log_raw
        )

    known = finite(
        r.get(
            "fall_onset_time_s"
        )
    )

    old_meta = metadata_onset(
        current
    )

    perturb, perturb_source = (
        perturb_from_log(
            log
        )
    )

    result = {
        "scenario_id": int(
            r["scenario_id"]
        ),
        "sim_age_profile": int(
            r["sim_age_profile"]
        ),
        "known_onset_s": known,
        "metadata_onset_s":
            old_meta,
        "perturbation_start_s":
            perturb,
        "perturbation_source":
            perturb_source,
        "candidate_onset_s":
            np.nan,
        "candidate_method":
            "",
        "known_error_s":
            np.nan,
        "known_abs_error_s":
            np.nan,
        "metadata_error_s":
            np.nan,
        "metadata_abs_error_s":
            np.nan,
        "h_ref_m":
            np.nan,
        "height_threshold_m":
            np.nan,
        "candidate_pelvis_height_m":
            np.nan,
        "candidate_trunk_lean_deg":
            np.nan,
        "candidate_horizontal_speed_mps":
            np.nan,
        "candidate_h_cond":
            False,
        "candidate_trunk_cond":
            False,
        "candidate_v_cond":
            False,
        "timeline_delta_s":
            np.nan,
        "status":
            "FAIL",
        "error":
            "",
    }

    try:
        if (
            markers is None
            or not markers.exists()
        ):
            raise RuntimeError(
                "markers CSV missing"
            )

        if (
            segments is None
            or not segments.exists()
        ):
            raise RuntimeError(
                "segments CSV missing"
            )

        rec = reconstruct_common_onset(
            markers,
            segments,
            perturb,
        )

        candidate = rec[
            "onset_s"
        ]

        result[
            "candidate_onset_s"
        ] = candidate

        result[
            "candidate_method"
        ] = rec[
            "method"
        ]

        result[
            "h_ref_m"
        ] = rec[
            "h_ref_m"
        ]

        result[
            "height_threshold_m"
        ] = rec[
            "height_threshold_m"
        ]

        result[
            "candidate_pelvis_height_m"
        ] = rec[
            "pelvis_height_m"
        ]

        result[
            "candidate_trunk_lean_deg"
        ] = rec[
            "trunk_lean_deg"
        ]

        result[
            "candidate_horizontal_speed_mps"
        ] = rec[
            "horizontal_speed_mps"
        ]

        result[
            "candidate_h_cond"
        ] = rec[
            "h_cond"
        ]

        result[
            "candidate_trunk_cond"
        ] = rec[
            "trunk_cond"
        ]

        result[
            "candidate_v_cond"
        ] = rec[
            "v_cond"
        ]

        result[
            "timeline_delta_s"
        ] = rec[
            "max_marker_segment_time_delta_s"
        ]

        if known is not None:
            err = (
                candidate
                - known
            )

            result[
                "known_error_s"
            ] = err

            result[
                "known_abs_error_s"
            ] = abs(err)

        if old_meta is not None:
            err = (
                candidate
                - old_meta
            )

            result[
                "metadata_error_s"
            ] = err

            result[
                "metadata_abs_error_s"
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
        "known_onset_s"
    ].notna()
    &
    (
        out["status"]
        == "OK"
    )
].copy()

missing = out[
    out[
        "known_onset_s"
    ].isna()
].copy()

task20 = out[
    out[
        "scenario_id"
    ] == 20
].copy()


print("=" * 100)
print("COMMON ONSET RECONSTRUCTION VALIDATION")
print("=" * 100)

print(
    "records total           :",
    len(out),
)

print(
    "reconstruction OK       :",
    int(
        (
            out.status
            == "OK"
        ).sum()
    ),
)

print(
    "reconstruction FAIL     :",
    int(
        (
            out.status
            != "OK"
        ).sum()
    ),
)

print(
    "known onset records     :",
    len(known),
)

print(
    "currently missing onset :",
    len(missing),
)


print()
print("Candidate methods:")
print(
    out[
        "candidate_method"
    ]
    .value_counts()
    .to_string()
)


print()
print("=" * 100)
print("221 KNOWN-ONSET CROSS-CHECK")
print("=" * 100)

if len(known):

    e = known[
        "known_abs_error_s"
    ]

    print(
        "median abs error [s]:",
        e.median(),
    )

    print(
        "mean abs error [s]  :",
        e.mean(),
    )

    print(
        "p95 abs error [s]   :",
        e.quantile(
            0.95
        ),
    )

    print(
        "max abs error [s]   :",
        e.max(),
    )

    for tol in [
        0.005,
        0.02,
        0.04,
        0.10,
    ]:
        print(
            f"within {tol:.3f}s:",
            f"{(e <= tol).mean():.3%}",
        )


print()
print("Error by known scenario:")

if len(known):
    print(
        known.groupby(
            "scenario_id"
        )[
            "known_abs_error_s"
        ]
        .agg(
            [
                "count",
                "median",
                "max",
            ]
        )
        .to_string()
    )


print()
print("=" * 100)
print("EARLY 117 CANDIDATE COVERAGE")
print("=" * 100)

print(
    "candidate onset available:",
    int(
        missing[
            "candidate_onset_s"
        ].notna().sum()
    ),
    "/",
    len(missing),
)

print()
print(
    missing.groupby(
        "scenario_id"
    )[
        "candidate_onset_s"
    ]
    .apply(
        lambda x:
        int(
            x.notna().sum()
        )
    )
    .to_string()
)


print()
print("=" * 100)
print("TASK 20 EXISTING METADATA vs COMMON-RULE CANDIDATE")
print("=" * 100)

print(
    task20[
        [
            "sim_age_profile",
            "metadata_onset_s",
            "candidate_onset_s",
            "metadata_abs_error_s",
            "perturbation_start_s",
            "h_ref_m",
            "height_threshold_m",
            "candidate_pelvis_height_m",
            "candidate_trunk_lean_deg",
            "candidate_horizontal_speed_mps",
            "candidate_method",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("EARLY ONSET->IMPACT FEASIBILITY")
print("=" * 100)

# Read impact candidates produced by previous validated audit.
impact_audit = (
    CAMPAIGN
    / "dynamics_impact_recovery_audit.csv"
)

if impact_audit.exists():

    imp = pd.read_csv(
        impact_audit
    )

    q = missing.merge(
        imp[
            [
                "scenario_id",
                "sim_age_profile",
                "candidate_impact_time_s",
            ]
        ],
        on=[
            "scenario_id",
            "sim_age_profile",
        ],
        how="left",
    )

    q[
        "candidate_onset_to_impact_s"
    ] = (
        q[
            "candidate_impact_time_s"
        ]
        -
        q[
            "candidate_onset_s"
        ]
    )

    print(
        q.groupby(
            "scenario_id"
        )[
            "candidate_onset_to_impact_s"
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

else:
    print(
        "previous impact audit not found"
    )


bad = out[
    out.status != "OK"
]

if len(bad):
    print(
        "\n===== RECONSTRUCTION FAILURES ====="
    )

    print(
        bad[
            [
                "scenario_id",
                "sim_age_profile",
                "error",
            ]
        ].to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# Validation gate.
#
# Saved marker/segment streams are at native ~30 Hz, while
# published/common event timestamps are rounded to milliseconds.
#
# Require:
#   338/338 reconstructable
#   exactly 221 known records tested
#   p95 <= one native frame + rounding
#   max <= two native frames
#   all 117 currently unresolved receive candidates
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
            "known_abs_error_s"
        ].quantile(
            0.95
        )
    ) <= 0.040
    and float(
        known[
            "known_abs_error_s"
        ].max()
    ) <= 0.070
    and int(
        missing[
            "candidate_onset_s"
        ].notna().sum()
    ) == 117
)


print()
print(
    "COMMON_ONSET_RULE_VALIDATED:",
    validation_ok,
)

if validation_ok:
    print(
        "The saved marker/segment streams reproduce "
        "the common onset detector sufficiently for "
        "offline recovery."
    )
    print(
        "Do not rerun the simulator."
    )
else:
    print(
        "Do not fill the missing onset timestamps yet."
    )
    print(
        "Review the scenario-level errors first."
    )


print()
print(
    "Audit CSV:",
    OUT,
)
