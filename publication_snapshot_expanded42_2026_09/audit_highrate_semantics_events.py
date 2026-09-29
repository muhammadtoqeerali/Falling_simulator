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
    / "outputs"
    / "_highrate_overnight"
    / "campaign_highrate_truth_v1"
)

MANIFEST = (
    CAMPAIGN
    / "highrate_pristine_truth_manifest.csv"
)

PROFILES = CAMPAIGN / "profiles_used.csv"

EXPECTED_TASKS = [
    20, 21, 22, 23, 24,
    250,
    25, 26, 27, 28, 29,
    290, 291,
    30, 31, 32, 33, 34,
    37, 38, 39, 40, 41, 42, 43, 44,
]


def read_csv(path):
    return pd.read_csv(
        path,
        comment="#",
    )


def read_comment_metadata(path):
    meta = {}

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            for line in f:
                if not line.startswith("#"):
                    break

                s = line[1:].strip()

                if ":" not in s:
                    continue

                k, v = s.split(":", 1)

                meta[
                    k.strip().lower()
                ] = v.strip()

    except Exception:
        pass

    return meta


def as_int(v):
    try:
        return int(
            float(str(v).strip())
        )
    except Exception:
        return None


def as_float(v):
    try:
        x = float(v)

        if math.isfinite(x):
            return x

    except Exception:
        pass

    return None


def recursive_values(obj, wanted):
    wanted = {
        str(x).lower()
        for x in wanted
    }

    found = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():

                if str(k).lower() in wanted:
                    found.append(v)

                walk(v)

        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)

    return found


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


def event_from_json(path):
    if not path.exists():
        return None

    obj = read_json(path)

    if obj is None:
        return None

    onset_vals = recursive_values(
        obj,
        {
            "fall_onset_time_s",
            "fall_onset_s",
            "onset_time_s",
        },
    )

    impact_vals = recursive_values(
        obj,
        {
            "main_impact_time_s",
            "impact_time_s",
            "fall_impact_time_s",
        },
    )

    onset = next(
        (
            as_float(x)
            for x in onset_vals
            if as_float(x) is not None
        ),
        None,
    )

    impact = next(
        (
            as_float(x)
            for x in impact_vals
            if as_float(x) is not None
        ),
        None,
    )

    if (
        onset is not None
        and impact is not None
    ):
        return onset, impact

    return None


def get_time_column(df):
    candidates = [
        "timestamp",
        "Time_s",
        "time",
        "t",
        "TimeStamp(s)",
    ]

    for c in candidates:
        if c in df.columns:
            x = pd.to_numeric(
                df[c],
                errors="coerce",
            )

            if x.notna().sum() >= 2:
                return c

    return None


def constant_time(
    df,
    candidates,
):
    for c in candidates:

        if c not in df.columns:
            continue

        x = pd.to_numeric(
            df[c],
            errors="coerce",
        ).dropna()

        if len(x):
            return float(
                x.iloc[0]
            )

    return None


def event_from_dataframe(df):
    onset = constant_time(
        df,
        [
            "Fall_Onset_Time_s",
            "fall_onset_time_s",
        ],
    )

    impact = constant_time(
        df,
        [
            "Main_Impact_Time_s",
            "main_impact_time_s",
        ],
    )

    if (
        onset is not None
        and impact is not None
    ):
        return onset, impact

    tc = get_time_column(df)

    if tc is None:
        return None

    t = pd.to_numeric(
        df[tc],
        errors="coerce",
    ).to_numpy(float)

    onset_idx = None
    impact_idx = None

    for c in [
        "PreImpact_Fall_Label",
        "Fall_Onset_Label",
    ]:
        if c in df.columns:

            z = (
                pd.to_numeric(
                    df[c],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(float)
            )

            hits = np.where(
                z > 0
            )[0]

            if len(hits):
                onset_idx = int(
                    hits[0]
                )
                break

    for c in [
        "Impact_Label",
        "Main_Impact_Label",
    ]:
        if c in df.columns:

            z = (
                pd.to_numeric(
                    df[c],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(float)
            )

            hits = np.where(
                z > 0
            )[0]

            if len(hits):
                impact_idx = int(
                    hits[0]
                )
                break

    if (
        onset_idx is not None
        and impact_idx is not None
    ):
        return (
            float(t[onset_idx]),
            float(t[impact_idx]),
        )

    return None


def resolve_scenario_id(
    legacy_csv,
    current_truth,
):
    # --------------------------------------------------------
    # 1. Legacy CSV embedded metadata is authoritative.
    # Particularly important because scenario250 writes a
    # fall_scenario25_* filename.
    # --------------------------------------------------------
    meta = read_comment_metadata(
        legacy_csv
    )

    for key in [
        "scenario_id",
        "task_id",
        "scenario",
    ]:
        if key in meta:

            v = as_int(
                meta[key]
            )

            if v is not None:
                return (
                    v,
                    f"legacy_comment:{key}",
                )

    # --------------------------------------------------------
    # 2. High-rate provenance metadata.
    # --------------------------------------------------------
    meta2 = read_comment_metadata(
        current_truth
    )

    for key in [
        "legacy_metadata_scenario_id",
        "legacy_metadata_task_id",
        "scenario_id",
        "task_id",
    ]:
        if key in meta2:

            v = as_int(
                meta2[key]
            )

            if v is not None:
                return (
                    v,
                    f"highrate_comment:{key}",
                )

    # --------------------------------------------------------
    # 3. Run manifest.
    # --------------------------------------------------------
    run_manifest = (
        current_truth.parent
        / "run_manifest.json"
    )

    obj = read_json(
        run_manifest
    )

    if obj is not None:

        vals = recursive_values(
            obj,
            {
                "scenario_id",
                "task_id",
            },
        )

        for x in vals:
            v = as_int(x)

            if v is not None:
                return (
                    v,
                    "run_manifest",
                )

    # --------------------------------------------------------
    # 4. Folder name last-resort fallback.
    # Never preferred because scenario250 may masquerade as 25.
    # --------------------------------------------------------
    m = re.search(
        r"scenario(\d+)_",
        current_truth.parent.name,
    )

    if m:
        return (
            int(m.group(1)),
            "folder_fallback",
        )

    return None, "unresolved"


def resolve_event(
    current_truth,
    legacy_csv,
):
    # --------------------------------------------------------
    # A. Current high-rate labeled CSV if present.
    # --------------------------------------------------------
    try:
        cur = read_csv(
            current_truth
        )

        ev = event_from_dataframe(
            cur
        )

        if ev is not None:
            return (
                ev[0],
                ev[1],
                "current_highrate_labels",
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # B. High-rate label metadata JSON.
    # --------------------------------------------------------
    highrate_meta = (
        current_truth.with_name(
            current_truth.stem
            + ".label_metadata.json"
        )
    )

    ev = event_from_json(
        highrate_meta
    )

    if ev is not None:
        return (
            ev[0],
            ev[1],
            "highrate_label_metadata",
        )

    # --------------------------------------------------------
    # C. Legacy label metadata JSON.
    # --------------------------------------------------------
    legacy_meta = (
        legacy_csv.with_name(
            legacy_csv.stem
            + ".label_metadata.json"
        )
    )

    ev = event_from_json(
        legacy_meta
    )

    if ev is not None:
        return (
            ev[0],
            ev[1],
            "legacy_label_metadata",
        )

    # --------------------------------------------------------
    # D. Legacy labeled CSV itself.
    # --------------------------------------------------------
    try:
        leg = read_csv(
            legacy_csv
        )

        ev = event_from_dataframe(
            leg
        )

        if ev is not None:
            return (
                ev[0],
                ev[1],
                "legacy_csv_labels",
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # E. Run manifest as final structured source.
    # --------------------------------------------------------
    run_manifest = (
        current_truth.parent
        / "run_manifest.json"
    )

    ev = event_from_json(
        run_manifest
    )

    if ev is not None:
        return (
            ev[0],
            ev[1],
            "run_manifest",
        )

    return None, None, "unresolved"


if not MANIFEST.exists():
    raise SystemExit(
        f"Missing manifest:\n{MANIFEST}"
    )


base = pd.read_csv(
    MANIFEST
)

profiles = {}

if PROFILES.exists():
    pp = pd.read_csv(
        PROFILES
    )

    for _, x in pp.iterrows():

        profiles[
            int(x["age"])
        ] = {
            "height": float(
                x["height"]
            ),
            "sex": str(
                x["sex"]
            ),
            "weight": float(
                x["weight"]
            ),
        }


rows = []


for _, r in base.iterrows():

    pristine = Path(
        r["pristine_truth_source"]
    )

    current = Path(
        r["current_truth_csv"]
    )

    legacy = Path(
        r["legacy_csv"]
    )

    if not pristine.is_absolute():
        pristine = ROOT / pristine

    if not current.is_absolute():
        current = ROOT / current

    if not legacy.is_absolute():
        legacy = ROOT / legacy

    scenario_id, scenario_source = (
        resolve_scenario_id(
            legacy,
            current,
        )
    )

    onset, impact, event_source = (
        resolve_event(
            current,
            legacy,
        )
    )

    age = int(
        r["age"]
    )

    profile = profiles.get(
        age,
        {},
    )

    result = {
        "sim_age_profile": age,
        "height_m": profile.get(
            "height",
            np.nan,
        ),
        "sex": profile.get(
            "sex",
            "",
        ),
        "weight_kg": profile.get(
            "weight",
            np.nan,
        ),
        "scenario_id": scenario_id,
        "scenario_id_source": scenario_source,
        "path_inferred_task": int(
            r["task"]
        ),
        "pristine_truth_source": str(
            pristine
        ),
        "current_truth_csv": str(
            current
        ),
        "legacy_csv": str(
            legacy
        ),
        "truth_source_type": r[
            "source_type"
        ],
        "fall_onset_time_s": onset,
        "main_impact_time_s": impact,
        "event_source": event_source,
        "event_valid": False,
        "validation_group": "",
        "status": "FAIL",
        "error": "",
    }

    try:
        if scenario_id is None:
            raise RuntimeError(
                "scenario ID unresolved"
            )

        truth = read_csv(
            pristine
        )

        t = pd.to_numeric(
            truth["timestamp"],
            errors="coerce",
        ).to_numpy(float)

        t0 = float(
            np.nanmin(t)
        )

        t1 = float(
            np.nanmax(t)
        )

        if onset is None:
            raise RuntimeError(
                "fall onset unresolved"
            )

        if impact is None:
            raise RuntimeError(
                "main impact unresolved"
            )

        if impact < onset:
            raise RuntimeError(
                f"impact {impact} < onset {onset}"
            )

        # small tolerance because event timestamps may come
        # from 30-Hz control timestamps while truth is 100 Hz.
        tol = 0.05

        if not (
            t0 - tol
            <= onset
            <= t1 + tol
        ):
            raise RuntimeError(
                f"onset outside truth range "
                f"[{t0}, {t1}]"
            )

        if not (
            t0 - tol
            <= impact
            <= t1 + tol
        ):
            raise RuntimeError(
                f"impact outside truth range "
                f"[{t0}, {t1}]"
            )

        result[
            "event_valid"
        ] = True

        if scenario_id in (
            list(range(20, 35))
            + [37, 38]
        ):
            result[
                "validation_group"
            ] = "primary_real_comparison"

        elif scenario_id in [39, 40]:
            result[
                "validation_group"
            ] = "protocol_review_height"

        elif scenario_id in [41, 42]:
            result[
                "validation_group"
            ] = "protocol_review_stairs_vs_ladder"

        else:
            result[
                "validation_group"
            ] = "simulation_extension"

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

# Assign trial numbers AFTER true semantic IDs have been recovered.
out = out.sort_values(
    [
        "sim_age_profile",
        "scenario_id",
        "current_truth_csv",
    ],
    na_position="last",
).reset_index(
    drop=True
)

out[
    "trial_id"
] = (
    out.groupby(
        [
            "sim_age_profile",
            "scenario_id",
        ],
        dropna=False,
    )
    .cumcount()
    + 1
)


OUTFILE = (
    CAMPAIGN
    / "highrate_semantic_event_manifest.csv"
)

out.to_csv(
    OUTFILE,
    index=False,
)


print("=" * 100)
print("HIGH-RATE SEMANTIC + EVENT AUDIT")
print("=" * 100)

print(
    "records:",
    len(out),
)

print(
    "OK:",
    int(
        (out.status == "OK").sum()
    ),
)

print(
    "FAIL:",
    int(
        (out.status != "OK").sum()
    ),
)

print()
print("TRUE semantic scenario counts:")
print(
    out.groupby(
        "scenario_id"
    )
    .size()
    .to_string()
)

print()
print("Scenario-ID source:")
print(
    out[
        "scenario_id_source"
    ]
    .value_counts()
    .to_string()
)

print()
print("Event source:")
print(
    out[
        "event_source"
    ]
    .value_counts()
    .to_string()
)

print()
print("Validation groups:")
print(
    out[
        "validation_group"
    ]
    .value_counts()
    .to_string()
)

print()
print(
    "Path says 25 but semantic task differs:"
)

alias = out[
    (
        out[
            "path_inferred_task"
        ] == 25
    )
    &
    (
        out[
            "scenario_id"
        ] != 25
    )
]

print(
    alias[
        [
            "sim_age_profile",
            "path_inferred_task",
            "scenario_id",
            "scenario_id_source",
            "current_truth_csv",
        ]
    ].to_string(
        index=False
    )
    if len(alias)
    else "NONE"
)

print()
print("Trials per age/task greater than 1:")

dups = (
    out.groupby(
        [
            "sim_age_profile",
            "scenario_id",
        ]
    )
    .size()
)

dups = dups[
    dups > 1
]

print(
    dups.to_string()
    if len(dups)
    else "NONE"
)


counts = (
    out.groupby(
        "scenario_id"
    )
    .size()
    .to_dict()
)

expected_ok = (
    len(out) == 338
    and set(
        counts.keys()
    ) == set(
        EXPECTED_TASKS
    )
    and all(
        counts.get(
            t,
            0,
        ) == 13
        for t in EXPECTED_TASKS
    )
)

events_ok = bool(
    (
        out.status
        == "OK"
    ).all()
)


print()
print(
    "SEMANTIC_COUNTS_OK :",
    expected_ok,
)

print(
    "EVENT_TIMING_OK    :",
    events_ok,
)

if (
    expected_ok
    and events_ok
):
    print(
        "\nSEMANTIC_EVENT_AUDIT_OK"
    )
else:
    print(
        "\nSEMANTIC_EVENT_AUDIT_NEEDS_REVIEW"
    )

    bad = out[
        out.status != "OK"
    ]

    if len(bad):
        print(
            "\nFailures:"
        )

        print(
            bad[
                [
                    "sim_age_profile",
                    "scenario_id",
                    "current_truth_csv",
                    "event_source",
                    "error",
                ]
            ].to_string(
                index=False
            )
        )

print()
print(
    "Manifest:",
    OUTFILE,
)
