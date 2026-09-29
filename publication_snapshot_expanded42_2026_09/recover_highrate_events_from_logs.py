from pathlib import Path
import re
import numpy as np
import pandas as pd


ROOT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
).resolve()

CAMPAIGN = (
    ROOT
    / "outputs/_highrate_overnight/"
    "campaign_highrate_truth_v1"
)

INPUT = (
    CAMPAIGN
    / "highrate_semantic_event_manifest.csv"
)

OUTPUT = (
    CAMPAIGN
    / "highrate_semantic_event_manifest_v2.csv"
)


def norm_path(x):
    p = Path(str(x))

    if not p.is_absolute():
        p = ROOT / p

    return p.resolve()


def strip_ansi(s):
    return re.sub(
        r"\x1b\[[0-9;]*[A-Za-z]",
        "",
        s,
    )


def parse_output_folder(text):
    matches = re.findall(
        r"Output folder\s*->\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None

    raw = strip_ansi(
        matches[-1]
    ).strip()

    return norm_path(raw)


def parse_common_event_summary(text):
    parts = re.split(
        r"COMMON FALL EVENT SUMMARY",
        text,
        flags=re.IGNORECASE,
    )

    if len(parts) < 2:
        return None

    block = parts[-1]

    # Do not silently accept an explicitly unavailable detector.
    if re.search(
        r"Event detector\s*:\s*unavailable",
        block,
        flags=re.IGNORECASE,
    ):
        return None

    def grab(pattern):
        m = re.search(
            pattern,
            block,
            flags=re.IGNORECASE,
        )

        if not m:
            return None

        try:
            return float(
                m.group(1)
            )
        except Exception:
            return None

    onset = grab(
        r"Fall\s+onset\s*:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*s"
    )

    impact = grab(
        r"Main\s+impact\s*:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*s"
    )

    perturb = grab(
        r"Perturb\s+start\s*:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*s"
    )

    settle = grab(
        r"Settle\s+time\s*:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*s"
    )

    if onset is None or impact is None:
        return None

    return {
        "fall_onset_time_s": onset,
        "main_impact_time_s": impact,
        "perturbation_start_time_s": perturb,
        "settle_time_s": settle,
    }


def validation_group(task):
    task = int(task)

    if 20 <= task <= 34:
        return "primary_real_comparison"

    if task in (37, 38):
        return "primary_real_comparison"

    if task in (39, 40):
        return "protocol_review_height"

    if task in (41, 42):
        return "protocol_review_stairs_vs_ladder"

    return "simulation_extension"


if not INPUT.exists():
    raise SystemExit(
        f"Missing input manifest:\n{INPUT}"
    )


print("=" * 100)
print("INDEXING SAVED BATCH LOGS")
print("=" * 100)

log_index = {}

logs = sorted(
    ROOT.glob(
        "outputs/batch_runs/"
        "batch_*/logs/scenario*_*.log"
    )
)

print("scenario logs found:", len(logs))

logs_with_folder = 0
logs_with_events = 0
duplicate_folders = []

for log in logs:
    try:
        text = log.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        continue

    folder = parse_output_folder(
        text
    )

    if folder is None:
        continue

    logs_with_folder += 1

    events = parse_common_event_summary(
        text
    )

    if events is not None:
        logs_with_events += 1

    key = str(folder)

    entry = {
        "log": str(log.resolve()),
        "events": events,
    }

    if key in log_index:
        duplicate_folders.append(
            key
        )

        # Prefer a log that actually contains event timing.
        old_has = (
            log_index[key]["events"]
            is not None
        )

        new_has = (
            events is not None
        )

        if new_has and not old_has:
            log_index[key] = entry

    else:
        log_index[key] = entry


print(
    "logs with output folder:",
    logs_with_folder,
)

print(
    "logs with event summary:",
    logs_with_events,
)

print(
    "unique output folders:",
    len(log_index),
)

print(
    "duplicate folder mappings:",
    len(set(duplicate_folders)),
)


manifest = pd.read_csv(
    INPUT
)

records = []

crosscheck = []


for _, row in manifest.iterrows():

    r = row.to_dict()

    current = norm_path(
        r["current_truth_csv"]
    )

    pristine = norm_path(
        r["pristine_truth_source"]
    )

    folder_key = str(
        current.parent
    )

    entry = log_index.get(
        folder_key
    )

    log_events = (
        entry["events"]
        if entry is not None
        else None
    )

    r["batch_log"] = (
        entry["log"]
        if entry is not None
        else ""
    )

    r["log_event_found"] = (
        log_events is not None
    )

    old_onset = pd.to_numeric(
        pd.Series([
            r.get(
                "fall_onset_time_s"
            )
        ]),
        errors="coerce",
    ).iloc[0]

    old_impact = pd.to_numeric(
        pd.Series([
            r.get(
                "main_impact_time_s"
            )
        ]),
        errors="coerce",
    ).iloc[0]

    old_valid = (
        np.isfinite(old_onset)
        and np.isfinite(old_impact)
    )

    # --------------------------------------------------------
    # Independent cross-check where both sources exist.
    # --------------------------------------------------------
    if (
        old_valid
        and log_events is not None
    ):
        d_onset = abs(
            float(old_onset)
            - float(
                log_events[
                    "fall_onset_time_s"
                ]
            )
        )

        d_impact = abs(
            float(old_impact)
            - float(
                log_events[
                    "main_impact_time_s"
                ]
            )
        )

        crosscheck.append({
            "scenario_id": int(
                r["scenario_id"]
            ),
            "sim_age_profile": int(
                r["sim_age_profile"]
            ),
            "onset_abs_diff_s": d_onset,
            "impact_abs_diff_s": d_impact,
        })

    # --------------------------------------------------------
    # Recover only unresolved events.
    # Never replace existing valid event annotations.
    # --------------------------------------------------------
    if (
        not old_valid
        and log_events is not None
    ):
        r[
            "fall_onset_time_s"
        ] = log_events[
            "fall_onset_time_s"
        ]

        r[
            "main_impact_time_s"
        ] = log_events[
            "main_impact_time_s"
        ]

        r[
            "event_source"
        ] = (
            "saved_batch_log_"
            "common_fall_event_summary"
        )

    # --------------------------------------------------------
    # Validate against pristine 100-Hz truth timeline.
    # --------------------------------------------------------
    r["status"] = "FAIL"
    r["error"] = ""
    r["event_valid"] = False

    try:
        onset = float(
            r["fall_onset_time_s"]
        )

        impact = float(
            r["main_impact_time_s"]
        )

        if not (
            np.isfinite(onset)
            and np.isfinite(impact)
        ):
            raise RuntimeError(
                "event timing unresolved"
            )

        if impact < onset:
            raise RuntimeError(
                f"impact {impact} < onset {onset}"
            )

        truth = pd.read_csv(
            pristine,
            comment="#",
        )

        t = pd.to_numeric(
            truth["timestamp"],
            errors="coerce",
        ).to_numpy(float)

        t = t[
            np.isfinite(t)
        ]

        if len(t) < 2:
            raise RuntimeError(
                "invalid truth timeline"
            )

        t0 = float(
            np.min(t)
        )

        t1 = float(
            np.max(t)
        )

        # Control/event timestamps are 30 Hz whereas the
        # canonical truth stream is resampled to 100 Hz.
        tol = 0.05

        if not (
            t0 - tol
            <= onset
            <= t1 + tol
        ):
            raise RuntimeError(
                f"onset {onset} outside "
                f"[{t0}, {t1}]"
            )

        if not (
            t0 - tol
            <= impact
            <= t1 + tol
        ):
            raise RuntimeError(
                f"impact {impact} outside "
                f"[{t0}, {t1}]"
            )

        r["event_valid"] = True
        r["status"] = "OK"

        r[
            "validation_group"
        ] = validation_group(
            r["scenario_id"]
        )

    except Exception as exc:
        r["error"] = str(exc)

    records.append(r)


out = pd.DataFrame(
    records
)

out.to_csv(
    OUTPUT,
    index=False,
)


cc = pd.DataFrame(
    crosscheck
)

CC_OUT = (
    CAMPAIGN
    / "event_log_crosscheck.csv"
)

cc.to_csv(
    CC_OUT,
    index=False,
)


print()
print("=" * 100)
print("EVENT RECOVERY RESULTS")
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
print("Event sources:")
print(
    out.event_source
    .value_counts(
        dropna=False
    )
    .to_string()
)

print()
print("Events recovered from saved logs:")
print(
    int(
        (
            out.event_source
            ==
            "saved_batch_log_common_fall_event_summary"
        ).sum()
    )
)

print()
print("Recovered log events by scenario:")
recovered = out[
    out.event_source
    ==
    "saved_batch_log_common_fall_event_summary"
]

print(
    recovered.groupby(
        "scenario_id"
    )
    .size()
    .to_string()
    if len(recovered)
    else "NONE"
)


print()
print("=" * 100)
print("INDEPENDENT LOG-vs-LABEL CROSS-CHECK")
print("=" * 100)

print(
    "records compared:",
    len(cc),
)

if len(cc):

    print(
        "max onset difference [s]:",
        cc[
            "onset_abs_diff_s"
        ].max(),
    )

    print(
        "max impact difference [s]:",
        cc[
            "impact_abs_diff_s"
        ].max(),
    )

    print(
        "median onset difference [s]:",
        cc[
            "onset_abs_diff_s"
        ].median(),
    )

    print(
        "median impact difference [s]:",
        cc[
            "impact_abs_diff_s"
        ].median(),
    )


counts = (
    out.groupby(
        "scenario_id"
    )
    .size()
)

semantic_ok = (
    len(out) == 338
    and len(counts) == 26
    and bool(
        (counts == 13).all()
    )
)

events_ok = bool(
    (out.status == "OK").all()
)

# Existing labels and saved stdout are both rounded to
# millisecond-scale values, so 5 ms is a conservative check.
crosscheck_ok = (
    len(cc) > 0
    and float(
        cc[
            "onset_abs_diff_s"
        ].max()
    ) <= 0.005
    and float(
        cc[
            "impact_abs_diff_s"
        ].max()
    ) <= 0.005
)


print()
print(
    "SEMANTIC_COUNTS_OK :",
    semantic_ok,
)

print(
    "ALL_EVENT_TIMING_OK:",
    events_ok,
)

print(
    "LOG_CROSSCHECK_OK  :",
    crosscheck_ok,
)


bad = out[
    out.status != "OK"
]

if len(bad):
    print("\n===== REMAINING FAILURES =====")

    print(
        bad[
            [
                "sim_age_profile",
                "scenario_id",
                "batch_log",
                "event_source",
                "error",
            ]
        ].to_string(
            index=False
        )
    )


if (
    semantic_ok
    and events_ok
    and crosscheck_ok
):
    print(
        "\nHIGH_RATE_EVENT_RECOVERY_OK"
    )

    print(
        "No simulator rerun is required."
    )

else:
    print(
        "\nHIGH_RATE_EVENT_RECOVERY_NEEDS_REVIEW"
    )


print()
print(
    "Final manifest:",
    OUTPUT,
)

print(
    "Cross-check CSV:",
    CC_OUT,
)
