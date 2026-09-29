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
    / "existing_onset_rules_audit.csv"
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


def first_json_value(obj, wanted):
    wanted = {
        str(x).lower()
        for x in wanted
    }

    hits = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():

                if str(k).lower() in wanted:
                    q = finite(v)

                    if q is not None:
                        hits.append(q)

                walk(v)

        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)

    return (
        hits[0]
        if hits
        else None
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

        return first_json_value(
            obj,
            {"fall_onset_time_s"},
        )

    except Exception:
        return None


def read_truth(path):
    return pd.read_csv(
        path,
        comment="#",
    )


def time_vector(df):
    for c in [
        "timestamp",
        "t",
        "time",
        "Time_s",
        "Time_s_standard",
    ]:
        if c in df.columns:

            x = pd.to_numeric(
                df[c],
                errors="coerce",
            ).to_numpy(float)

            if (
                np.isfinite(x).sum()
                >= 2
            ):
                return x

    return None


def first_existing_fall_flag(
    legacy_csv,
):
    """
    Reproduce the batch_run_all_labeled.py
    fallback at lines ~592-599:
      first fall_detected/fall > 0.
    """

    try:
        df = pd.read_csv(
            legacy_csv,
            comment="#",
        )
    except Exception:
        return None

    t = time_vector(df)

    if t is None:
        return None

    col = None

    for c in [
        "fall_detected",
        "fall",
    ]:
        if c in df.columns:
            col = c
            break

    if col is None:
        return None

    z = (
        pd.to_numeric(
            df[col],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(float)
    )

    hits = np.where(
        z > 0
    )[0]

    if not len(hits):
        return None

    return float(
        t[int(hits[0])]
    )


def pelvis_threshold_onset(
    pristine_csv,
    perturb,
):
    """
    Exact rule shown in label_imu_csv.py:

      pre = [perturb-2s, perturb)
      h_ref = max pelvis_height(pre)
      drop = max(0.20,
                 0.125 * max(h_ref, 1.0))
      threshold = h_ref - drop

      onset = first post-perturb sample
              pelvis_height <= threshold
    """

    if perturb is None:
        return None, None, None

    try:
        df = read_truth(
            pristine_csv
        )
    except Exception:
        return None, None, None

    if (
        "timestamp" not in df.columns
        or "pelvis_height" not in df.columns
    ):
        return None, None, None

    t = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    ).to_numpy(float)

    h = pd.to_numeric(
        df["pelvis_height"],
        errors="coerce",
    ).to_numpy(float)

    pre = (
        (t >= max(
            0.0,
            perturb - 2.0,
        ))
        &
        (t < perturb)
    )

    post = (
        t >= perturb
    )

    if (
        not np.any(pre)
        or not np.any(post)
    ):
        return None, None, None

    vals = h[
        pre
        &
        np.isfinite(h)
    ]

    if not len(vals):
        return None, None, None

    h_ref = float(
        np.nanmax(vals)
    )

    drop = max(
        0.20,
        0.125
        * max(
            h_ref,
            1.0,
        ),
    )

    threshold = (
        h_ref
        - drop
    )

    hits = np.where(
        post
        &
        np.isfinite(h)
        &
        (h <= threshold)
    )[0]

    if not len(hits):
        return (
            None,
            h_ref,
            threshold,
        )

    onset = float(
        t[int(hits[0])]
    )

    return (
        onset,
        h_ref,
        threshold,
    )


def error(candidate, known):
    if (
        candidate is None
        or known is None
    ):
        return np.nan

    return abs(
        float(candidate)
        - float(known)
    )


m = pd.read_csv(
    MANIFEST
)

rows = []


for _, r in m.iterrows():

    current = norm_path(
        r["current_truth_csv"]
    )

    pristine = norm_path(
        r["pristine_truth_source"]
    )

    legacy = norm_path(
        r["legacy_csv"]
    )

    log_raw = r.get(
        "batch_log",
        "",
    )

    log = (
        norm_path(log_raw)
        if (
            isinstance(log_raw, str)
            and log_raw.strip()
        )
        else None
    )

    known = finite(
        r.get(
            "fall_onset_time_s"
        )
    )

    meta_onset = (
        metadata_onset(
            current
        )
    )

    legacy_events = {}

    if (
        log is not None
        and log.exists()
    ):
        text = log.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        try:
            legacy_events = (
                batch
                .parse_legacy_phase_event_summary(
                    text
                )
                or {}
            )
        except Exception:
            legacy_events = {}

    log_fall = finite(
        legacy_events.get(
            "fall_onset_time_s"
        )
    )

    perturb = finite(
        legacy_events.get(
            "perturbation_start_time_s"
        )
    )

    csv_fall = (
        first_existing_fall_flag(
            legacy
        )
    )

    pelvis_fall, h_ref, threshold = (
        pelvis_threshold_onset(
            pristine,
            perturb,
        )
    )

    rows.append({
        "scenario_id": int(
            r["scenario_id"]
        ),
        "sim_age_profile": int(
            r["sim_age_profile"]
        ),

        "known_onset_s": known,

        "metadata_onset_s":
            meta_onset,

        "legacy_log_first_FALL_s":
            log_fall,

        "legacy_csv_first_fall_flag_s":
            csv_fall,

        "legacy_log_perturb_s":
            perturb,

        "label_imu_pelvis_onset_s":
            pelvis_fall,

        "pelvis_h_ref_m":
            h_ref,

        "pelvis_threshold_m":
            threshold,

        "err_log_first_FALL_s":
            error(
                log_fall,
                known,
            ),

        "err_csv_fall_flag_s":
            error(
                csv_fall,
                known,
            ),

        "err_pelvis_rule_s":
            error(
                pelvis_fall,
                known,
            ),

        "err_metadata_s":
            error(
                meta_onset,
                known,
            ),
    })


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
].copy()

missing = out[
    out[
        "known_onset_s"
    ].isna()
].copy()


print("=" * 100)
print("EXISTING ONSET RULE AUDIT")
print("=" * 100)

print(
    "records total:",
    len(out),
)

print(
    "known authoritative onsets:",
    len(known),
)

print(
    "currently unresolved:",
    len(missing),
)


def report(
    name,
    col,
    err_col,
):
    print()
    print("-" * 80)
    print(name)
    print("-" * 80)

    print(
        "known coverage:",
        int(
            known[col]
            .notna()
            .sum()
        ),
        "/",
        len(known),
    )

    print(
        "missing-record coverage:",
        int(
            missing[col]
            .notna()
            .sum()
        ),
        "/",
        len(missing),
    )

    e = known[
        err_col
    ].dropna()

    if len(e):
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
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
        ]:
            print(
                f"within {tol:.2f}s:",
                f"{(e <= tol).mean():.2%}",
            )


report(
    "A) Existing batch legacy log: first FALL! / native_hz",
    "legacy_log_first_FALL_s",
    "err_log_first_FALL_s",
)

report(
    "B) Existing batch CSV fallback: first fall/fall_detected flag",
    "legacy_csv_first_fall_flag_s",
    "err_csv_fall_flag_s",
)

report(
    "C) Existing label_imu_csv pelvis-height rule",
    "label_imu_pelvis_onset_s",
    "err_pelvis_rule_s",
)


print()
print("=" * 100)
print("COVERAGE OF 117 CURRENTLY UNRESOLVED RECORDS")
print("=" * 100)

for c in [
    "metadata_onset_s",
    "legacy_log_first_FALL_s",
    "legacy_csv_first_fall_flag_s",
    "legacy_log_perturb_s",
    "label_imu_pelvis_onset_s",
]:
    print()
    print(c)

    print(
        missing.groupby(
            "scenario_id"
        )[c]
        .apply(
            lambda x:
            int(x.notna().sum())
        )
        .to_string()
    )


print()
print("=" * 100)
print("TASK 20: DIFFERENCE BETWEEN EXISTING ONSET DEFINITIONS")
print("=" * 100)

t20 = out[
    out[
        "scenario_id"
    ] == 20
]

print(
    t20[
        [
            "sim_age_profile",
            "metadata_onset_s",
            "legacy_log_first_FALL_s",
            "legacy_csv_first_fall_flag_s",
            "legacy_log_perturb_s",
            "label_imu_pelvis_onset_s",
            "pelvis_h_ref_m",
            "pelvis_threshold_m",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("KNOWN-ONSET ERROR BY SCENARIO")
print("=" * 100)

for err_col in [
    "err_log_first_FALL_s",
    "err_csv_fall_flag_s",
    "err_pelvis_rule_s",
]:
    print()
    print(err_col)

    q = (
        known.groupby(
            "scenario_id"
        )[err_col]
        .agg(
            [
                "count",
                "median",
                "max",
            ]
        )
    )

    print(
        q.to_string()
    )


print()
print(
    "Audit CSV:",
    OUT,
)

print()
print(
    "ONSET_RULE_AUDIT_COMPLETE"
)
