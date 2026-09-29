#!/usr/bin/env python3

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


PROJECT = Path(
    "/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
)

CORRECTED = (
    PROJECT
    / "outputs/_highrate_overnight/"
      "campaign_highrate_truth_v3_rne_corrected396"
)

RUNS = CORRECTED / "runs"

PUB = (
    PROJECT
    / "outputs/phase2_publication_gate_audit_20260826"
)

OUT = (
    PROJECT
    / "outputs/paper_evidence_archive_v1/"
      "13_CANONICAL_PAPER_EVIDENCE_V3/"
      "compact_canonical_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# Helpers
# =====================================================================

def sim_csv(path, nrows=None):
    """
    Correct simulator CSV reader.

    The simulator files contain metadata lines beginning with '#'
    before the real CSV header.
    """
    return pd.read_csv(
        path,
        comment="#",
        nrows=nrows,
        low_memory=False,
    )


def parse_path(path):
    s = str(path)

    profile = re.search(
        r"/(P\d{3})/",
        s,
    )

    task = re.search(
        r"/task_(\d+)/",
        s,
    )

    age = re.search(
        r"(?:^|[_/])age(\d{1,3})(?:[_/]|$)",
        s,
    )

    sex = re.search(
        r"(?:^|[_/])sex_(male|female)(?:[_/]|$)",
        s,
        re.I,
    )

    return {
        "profile":
            profile.group(1)
            if profile else "",

        "task":
            int(task.group(1))
            if task else np.nan,

        "age":
            int(age.group(1))
            if age else np.nan,

        "sex":
            sex.group(1).lower()
            if sex else "",
    }


print("=" * 105)
print("COMPACT CANONICAL SIMULATOR VALIDATION AUDIT")
print("=" * 105)

print()
print("Corrected396 root :", CORRECTED)
print("Runs root         :", RUNS)
print("Runs root exists  :", RUNS.exists())


if not RUNS.exists():
    raise SystemExit(
        "ERROR: corrected396/runs directory does not exist."
    )


# =====================================================================
# 1. Recursive discovery
# =====================================================================

all_csv = sorted(
    p for p in RUNS.rglob("*.csv")
    if p.is_file()
)

truth_csv = [
    p for p in all_csv
    if p.name.endswith(
        "_highrate_truth.csv"
    )
]

standard_csv = [
    p for p in all_csv
    if not p.name.endswith(
        "_highrate_truth.csv"
    )
]

validation_txt = sorted(
    p for p in RUNS.rglob(
        "*_validation.txt"
    )
    if p.is_file()
)


print()
print("-" * 105)
print("RECURSIVE FILE DISCOVERY")
print("-" * 105)

print(
    "All CSV files               :",
    len(all_csv),
)

print(
    "Standard simulator CSV files:",
    len(standard_csv),
)

print(
    "High-rate truth CSV files   :",
    len(truth_csv),
)

print(
    "Validation reports          :",
    len(validation_txt),
)


if all_csv:
    print(
        "Example CSV:",
        all_csv[0],
    )

if validation_txt:
    print(
        "Example validation report:",
        validation_txt[0],
    )


if len(all_csv) == 0:
    raise SystemExit(
        "ERROR: recursive discovery still found zero CSV files."
    )


# =====================================================================
# 2. Schema audit
# =====================================================================

def collect_schemas(paths):

    schemas = {}
    errors = []

    for p in paths:

        try:
            d = sim_csv(
                p,
                nrows=2,
            )

            cols = tuple(
                str(c)
                for c in d.columns
            )

            if cols not in schemas:
                schemas[cols] = {
                    "count": 0,
                    "example": str(p),
                }

            schemas[cols][
                "count"
            ] += 1

        except Exception as e:

            errors.append({
                "file": str(p),
                "error": repr(e),
            })

    return schemas, errors


std_schemas, std_errors = (
    collect_schemas(
        standard_csv
    )
)

truth_schemas, truth_errors = (
    collect_schemas(
        truth_csv
    )
)


def show_schemas(
    title,
    schemas,
    errors,
):

    print()
    print("-" * 105)
    print(title)
    print("-" * 105)

    print(
        "Unique schemas:",
        len(schemas),
    )

    print(
        "Read errors:",
        len(errors),
    )

    for i, (
        cols,
        info,
    ) in enumerate(
        sorted(
            schemas.items(),
            key=lambda x:
                -x[1]["count"],
        )[:5],
        1,
    ):

        print()
        print(
            f"Schema {i}: "
            f"{info['count']} files"
        )

        print(
            "Columns:"
        )

        for c in cols:
            print(
                " ",
                c,
            )

        print(
            "Example:",
            info["example"],
        )

    if len(schemas) > 5:
        print()
        print(
            "...",
            len(schemas) - 5,
            "additional uncommon schemas "
            "saved to the output tables."
        )

    if errors:
        print()
        print(
            "First read error:"
        )

        print(
            errors[0]
        )


show_schemas(
    "STANDARD SIMULATOR CSV SCHEMAS",
    std_schemas,
    std_errors,
)

show_schemas(
    "HIGH-RATE PHYSICS-TRUTH CSV SCHEMAS",
    truth_schemas,
    truth_errors,
)


# Save complete schema audit without printing all of it.
schema_rows = []

for source_type, schemas in [
    (
        "standard",
        std_schemas,
    ),
    (
        "highrate_truth",
        truth_schemas,
    ),
]:

    for cols, info in schemas.items():

        schema_rows.append({
            "source_type":
                source_type,

            "file_count":
                info["count"],

            "columns":
                "|".join(cols),

            "example_file":
                info["example"],
        })


pd.DataFrame(
    schema_rows
).to_csv(
    OUT
    / "csv_schema_summary.csv",
    index=False,
)


pd.DataFrame(
    std_errors
).to_csv(
    OUT
    / "standard_csv_read_errors.csv",
    index=False,
)


pd.DataFrame(
    truth_errors
).to_csv(
    OUT
    / "highrate_truth_read_errors.csv",
    index=False,
)


# =====================================================================
# 3. Representative corrected396 trials
#
# Deterministic selection by age/task, NOT by validation score.
# =====================================================================

trial_rows = []

for p in standard_csv:

    meta = parse_path(p)

    if pd.isna(
        meta["age"]
    ):
        continue

    trial_rows.append({
        **meta,
        "file": str(p),
    })


trials = pd.DataFrame(
    trial_rows
)


print()
print("-" * 105)
print("CORRECTED396 PROFILE COVERAGE")
print("-" * 105)

if trials.empty:

    print(
        "No age/profile metadata could "
        "be parsed from standard CSV paths."
    )

else:

    print(
        "Trial files with parsed age:",
        len(trials),
    )

    print(
        "Profiles:",
        trials[
            "profile"
        ].nunique(),
    )

    print(
        "Tasks:",
        trials[
            "task"
        ].nunique(),
    )

    print(
        "Age range:",
        int(
            trials["age"].min()
        ),
        "–",
        int(
            trials["age"].max()
        ),
    )

    print(
        "Sex counts:"
    )

    print(
        trials[
            "sex"
        ]
        .value_counts()
        .to_string()
    )


targets = [
    20,
    30,
    53,
    65,
    74,
    78,
]

preferred_tasks = [
    21,
    29,
    32,
    40,
    42,
]

rep_rows = []


if not trials.empty:

    for target in targets:

        x = trials.copy()

        x[
            "age_distance"
        ] = (
            x["age"]
            - target
        ).abs()

        x[
            "task_rank"
        ] = (
            x["task"]
            .map({
                t: i
                for i, t in enumerate(
                    preferred_tasks
                )
            })
            .fillna(999)
        )

        x = x.sort_values([
            "age_distance",
            "task_rank",
            "profile",
            "task",
            "file",
        ])

        r = x.iloc[0]

        p = Path(
            r["file"]
        )

        try:
            d = sim_csv(
                p,
                nrows=3,
            )

            rep_rows.append({
                "target_age":
                    target,

                "actual_age":
                    int(
                        r["age"]
                    ),

                "profile":
                    r["profile"],

                "sex":
                    r["sex"],

                "task":
                    int(
                        r["task"]
                    ),

                "file":
                    str(p),

                "columns":
                    "|".join(
                        str(c)
                        for c in d.columns
                    ),
            })

        except Exception as e:

            rep_rows.append({
                "target_age":
                    target,

                "actual_age":
                    int(
                        r["age"]
                    ),

                "profile":
                    r["profile"],

                "sex":
                    r["sex"],

                "task":
                    int(
                        r["task"]
                    ),

                "file":
                    str(p),

                "columns":
                    "READ_ERROR:"
                    + repr(e),
            })


rep = pd.DataFrame(
    rep_rows
)

rep.to_csv(
    OUT
    / "representative_corrected396_trials.csv",
    index=False,
)


print()
print("-" * 105)
print("REPRESENTATIVE CORRECTED396 TRIALS")
print("-" * 105)

if rep.empty:

    print(
        "No representatives available."
    )

else:

    for r in rep.itertuples():

        print(
            f"age {r.actual_age:>2} | "
            f"{r.profile:<4} | "
            f"{r.sex:<6} | "
            f"task {r.task}"
        )

        print(
            " ",
            r.file,
        )


# =====================================================================
# 4. Parse all validation reports
# =====================================================================

checks = [
    "head_velocity_safe",
    "duration_realistic",
    "jerk_realistic",
    "protective_response",
    "imu_quality",
    "contact_pattern",
]

validation_rows = []


for p in validation_txt:

    text = p.read_text(
        encoding="utf-8",
        errors="replace",
    )

    meta = parse_path(
        p
    )

    conf = re.search(
        r"Overall Confidence:\s*"
        r"([0-9.]+)%",
        text,
    )

    classification = re.search(
        r"Classification:\s*"
        r"([A-Z_]+)",
        text,
    )

    row = {
        **meta,

        "confidence":
            float(
                conf.group(1)
            )
            if conf
            else np.nan,

        "classification":
            classification.group(1)
            if classification
            else "",

        "file":
            str(p),
    }

    for check in checks:

        m = re.search(
            rf"{re.escape(check)}"
            r"\s+(PASS|FAIL)",
            text,
        )

        row[check] = (
            m.group(1)
            if m
            else ""
        )

    validation_rows.append(
        row
    )


vr = pd.DataFrame(
    validation_rows,
    columns=[
        "profile",
        "task",
        "age",
        "sex",
        "confidence",
        "classification",
        *checks,
        "file",
    ],
)


vr.to_csv(
    OUT
    / "corrected396_validation_reports.csv",
    index=False,
)


print()
print("-" * 105)
print("CORRECTED396 VALIDATION SUMMARY")
print("-" * 105)

print(
    "Validation reports:",
    len(vr),
)


if len(vr) > 0:

    print(
        "Profiles:",
        vr[
            "profile"
        ].nunique(),
    )

    print(
        "Tasks:",
        vr[
            "task"
        ].nunique(),
    )

    conf = pd.to_numeric(
        vr["confidence"],
        errors="coerce",
    )

    valid_conf = conf.dropna()

    if len(valid_conf):

        print(
            "Confidence mean   :",
            f"{valid_conf.mean():.2f}%",
        )

        print(
            "Confidence median :",
            f"{valid_conf.median():.2f}%",
        )

        print(
            "Confidence range  :",
            f"{valid_conf.min():.2f}%"
            " – "
            f"{valid_conf.max():.2f}%",
        )

    print()
    print(
        "Classification counts:"
    )

    print(
        vr[
            "classification"
        ]
        .replace(
            "",
            "UNPARSED",
        )
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Individual check pass rates:"
    )

    pass_rows = []

    for check in checks:

        valid = vr[
            vr[check]
            .isin(
                [
                    "PASS",
                    "FAIL",
                ]
            )
        ]

        if len(valid) == 0:
            continue

        passed = int(
            valid[
                check
            ]
            .eq("PASS")
            .sum()
        )

        total = len(
            valid
        )

        rate = (
            100.0
            * passed
            / total
        )

        print(
            f"  {check:28s}"
            f"{rate:7.2f}% "
            f"({passed}/{total})"
        )

        pass_rows.append({
            "check":
                check,

            "pass_count":
                passed,

            "total":
                total,

            "pass_rate_percent":
                rate,
        })


    pd.DataFrame(
        pass_rows
    ).to_csv(
        OUT
        / "validation_check_pass_rates.csv",
        index=False,
    )


    # Per-task aggregate
    task_summary = (
        vr.groupby(
            "task",
            dropna=False,
        )
        .agg(
            trials=(
                "file",
                "count",
            ),

            confidence_mean=(
                "confidence",
                "mean",
            ),

            confidence_std=(
                "confidence",
                "std",
            ),

            confidence_min=(
                "confidence",
                "min",
            ),

            confidence_max=(
                "confidence",
                "max",
            ),
        )
        .reset_index()
    )


    for check in checks:

        tmp = (
            vr.assign(
                _pass=
                    vr[check]
                    .eq("PASS")
                    .astype(float)
            )
            .groupby(
                "task"
            )["_pass"]
            .mean()
            .mul(100.0)
            .rename(
                f"{check}_pass_pct"
            )
        )

        task_summary = (
            task_summary.merge(
                tmp,
                on="task",
                how="left",
            )
        )


    task_summary.to_csv(
        OUT
        / "corrected396_validation_by_task.csv",
        index=False,
    )


else:

    print(
        "ERROR: no validation reports parsed."
    )


# =====================================================================
# 5. Compact publication-gate inventory
# =====================================================================

aggregate_rows = []

keywords = [
    "summary",
    "metric",
    "audit",
    "event",
    "impact",
    "manifest",
    "validation",
    "comparison",
    "coverage",
    "count",
    "distribution",
]


for p in sorted(
    PUB.rglob("*")
):

    if not p.is_file():
        continue

    if p.suffix.lower() not in {
        ".csv",
        ".json",
        ".txt",
    }:
        continue

    # Do not include individual raw corrected396 trial files.
    if "campaign_highrate_truth_v3_rne_corrected396/runs" in str(p):
        continue

    if not any(
        k in p.name.lower()
        for k in keywords
    ):
        continue

    row = {
        "file":
            str(p),

        "type":
            p.suffix.lower(),

        "columns_or_keys":
            "",
    }

    if p.suffix.lower() == ".csv":

        try:
            d = pd.read_csv(
                p,
                comment="#",
                nrows=2,
                low_memory=False,
            )

            row[
                "columns_or_keys"
            ] = "|".join(
                str(c)
                for c in d.columns
            )

        except Exception as e:

            row[
                "columns_or_keys"
            ] = (
                "READ_ERROR:"
                + repr(e)
            )

    elif p.suffix.lower() == ".json":

        try:
            obj = json.loads(
                p.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

            if isinstance(
                obj,
                dict,
            ):
                row[
                    "columns_or_keys"
                ] = "|".join(
                    str(k)
                    for k in list(
                        obj.keys()
                    )[:40]
                )

        except Exception as e:

            row[
                "columns_or_keys"
            ] = (
                "READ_ERROR:"
                + repr(e)
            )

    aggregate_rows.append(
        row
    )


agg = pd.DataFrame(
    aggregate_rows
)

agg.to_csv(
    OUT
    / "publication_gate_aggregate_file_inventory.csv",
    index=False,
)


print()
print("-" * 105)
print("PUBLICATION-GATE AGGREGATE EVIDENCE")
print("-" * 105)

print(
    "Candidate aggregate files:",
    len(agg),
)


if len(agg):

    print()
    print(
        "First 20 candidate files:"
    )

    for p in (
        agg["file"]
        .head(20)
    ):

        print(
            " ",
            p,
        )

    if len(agg) > 20:

        print(
            " ... plus",
            len(agg) - 20,
            "additional files "
            "saved in the CSV inventory.",
        )


# =====================================================================
# 6. Compact summary
# =====================================================================

summary = {
    "all_csv_files":
        len(all_csv),

    "standard_csv_files":
        len(standard_csv),

    "highrate_truth_csv_files":
        len(truth_csv),

    "standard_unique_schemas":
        len(std_schemas),

    "highrate_truth_unique_schemas":
        len(truth_schemas),

    "standard_read_errors":
        len(std_errors),

    "highrate_truth_read_errors":
        len(truth_errors),

    "validation_reports":
        len(vr),

    "profiles":
        (
            int(
                vr["profile"]
                .nunique()
            )
            if len(vr)
            else 0
        ),

    "tasks":
        (
            int(
                vr["task"]
                .nunique()
            )
            if len(vr)
            else 0
        ),

    "publication_gate_aggregate_files":
        len(agg),
}


(
    OUT
    / "compact_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 105)

if (
    len(all_csv) > 0
    and len(validation_txt) > 0
):

    print(
        "COMPACT CANONICAL AUDIT: PASS"
    )

else:

    print(
        "COMPACT CANONICAL AUDIT: FAIL"
    )

print("=" * 105)

print(
    "Saved:",
    OUT,
)
