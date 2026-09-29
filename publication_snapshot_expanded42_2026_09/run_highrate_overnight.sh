#!/usr/bin/env bash

set -u

ROOT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
cd "$ROOT" || exit 1

# ------------------------------------------------------------
# Prevent normal suspend/idle sleep while this job is running.
# Re-exec ourselves under systemd-inhibit once.
# ------------------------------------------------------------
if command -v systemd-inhibit >/dev/null 2>&1 && \
   [ "${PROTECHTO_INHIBITED:-0}" != "1" ]; then

    export PROTECHTO_INHIBITED=1

    exec systemd-inhibit \
        --what=sleep:idle \
        --mode=block \
        --why="Protechto full high-rate fall simulation batch" \
        bash "$0"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUNROOT="outputs/_highrate_overnight/${STAMP}"

mkdir -p "$RUNROOT"

echo "$STAMP" > outputs/_highrate_overnight/LATEST_RUN
date +%s > "$RUNROOT/start_epoch.txt"
touch "$RUNROOT/start.marker"

LOG="$RUNROOT/batch.log"

exec > >(tee -a "$LOG") 2>&1

echo "================================================================================"
echo "PROTECHTO FULL HIGH-RATE OVERNIGHT RUN"
echo "================================================================================"
echo "Started       : $(date -Is)"
echo "Project       : $ROOT"
echo "Run directory : $RUNROOT"
echo "Python        : $ROOT/.venv/bin/python"
echo "PID           : $$"
echo

# ------------------------------------------------------------
# Runtime environment
# ------------------------------------------------------------
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

echo "===== PYTHON / TORCH / GPU PREFLIGHT ====="
./.venv/bin/python - <<'PY'
import sys
import torch

print("python executable     :", sys.executable)
print("torch version         :", torch.__version__)
print("torch cuda available  :", torch.cuda.is_available())
print("torch cuda count      :", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"cuda device {i}       :", torch.cuda.get_device_name(i))
PY

echo
echo "===== NVIDIA-SMI ====="
nvidia-smi || true

echo
echo "===== PATCH PREFLIGHT ====="

./.venv/bin/python -m py_compile \
    high_rate_imu_sidecar.py \
    highrate_runtime_bridge.py \
    fall_core.py \
    backward_fall_walking_best.py \
    scenario*_legacy.py

if [ $? -ne 0 ]; then
    echo "FATAL: Python compile preflight failed."
    exit 20
fi

echo "Compile preflight: OK"

PATCHED_COUNT="$(
    grep -l \
      'highrate_env_step(env, action, legacy_imu=imu)' \
      fall_core.py \
      backward_fall_walking_best.py \
      scenario*_legacy.py \
      2>/dev/null | wc -l
)"

echo "Patched runtime count: $PATCHED_COUNT"

if [ "$PATCHED_COUNT" -ne 27 ]; then
    echo "FATAL: expected 27 patched runtimes."
    exit 21
fi

echo
echo "===== EXISTING HIGH-RATE FILE COUNT ====="

BEFORE_HR="$(
    find outputs \
      -type f \
      -name '*_highrate_truth.csv' \
      2>/dev/null | wc -l
)"

echo "Before batch: $BEFORE_HR"

echo
echo "================================================================================"
echo "STARTING FULL BATCH"
echo "================================================================================"
echo

# ------------------------------------------------------------
# Full non-video batch.
# Do not delete any historical output.
# ------------------------------------------------------------
./.venv/bin/python - <<'PYPROFILES'
import pandas as pd
from pathlib import Path

src = Path(
    "outputs/simulated_dataset_all_subjects/"
    "conversion_reports/conversion_report_all_subjects.csv"
)

if not src.exists():
    raise SystemExit(f"Missing previous profile report: {src}")

r = pd.read_csv(src)

needed = ["age", "height_token", "sex", "weight_token"]

missing = [c for c in needed if c not in r.columns]

if missing:
    raise SystemExit(f"Missing profile columns: {missing}")

q = (
    r[needed]
    .dropna()
    .drop_duplicates()
    .copy()
)

q["height"] = (
    q["height_token"]
    .astype(str)
    .str.replace("p", ".", regex=False)
    .astype(float)
)

q["weight"] = (
    q["weight_token"]
    .astype(str)
    .str.replace("p", ".", regex=False)
    .astype(float)
)

profiles = (
    q[["age", "height", "sex", "weight"]]
    .drop_duplicates()
    .sort_values("age")
)

print("Profiles recovered:")
print(profiles.to_string(index=False))

if len(profiles) != 13:
    raise SystemExit(
        f"Expected 13 simulator profiles, found {len(profiles)}"
    )

out = Path("'"$RUNROOT"'") / "profiles.tsv"

profiles.to_csv(
    out,
    sep="\t",
    index=False,
    header=False,
)

print(f"Saved {len(profiles)} profiles -> {out}")
PYPROFILES

PROFILE_BUILD_RC=$?

if [ "$PROFILE_BUILD_RC" -ne 0 ]; then
    echo "FATAL: could not build simulator profile list."
    false
else
    BATCH_RC_ALL=0

    while IFS=$'\t' read -r AGE HEIGHT SEX WEIGHT
    do
        [ -z "$AGE" ] && continue

        MARK="$RUNROOT/profile_${AGE}.done"

        echo
        echo "================================================================================"
        echo "PROFILE AGE=$AGE HEIGHT=$HEIGHT SEX=$SEX WEIGHT=$WEIGHT"
        echo "================================================================================"

        if [ -f "$MARK" ]; then
            echo "Profile already completed; skipping."
            continue
        fi

        ./.venv/bin/python -u batch_run_all_labeled.py \
            --age "$AGE" \
            --height "$HEIGHT" \
            --sex "$SEX" \
            --weight "$WEIGHT" \
            --continue-on-error

        RC=$?

        echo "Profile age $AGE exit code: $RC"

        if [ "$RC" -eq 0 ]; then
            touch "$MARK"
        else
            BATCH_RC_ALL=1
            echo "$RC" > "$RUNROOT/profile_${AGE}.failed"
        fi

    done < "$RUNROOT/profiles.tsv"

    if [ "$BATCH_RC_ALL" -eq 0 ]; then
        true
    else
        false
    fi
fi

BATCH_RC=$?

echo
echo "================================================================================"
echo "BATCH PROCESS FINISHED"
echo "================================================================================"
echo "Exit code : $BATCH_RC"
echo "Finished  : $(date -Is)"

AFTER_HR="$(
    find outputs \
      -type f \
      -name '*_highrate_truth.csv' \
      2>/dev/null | wc -l
)"

echo "High-rate files before : $BEFORE_HR"
echo "High-rate files after  : $AFTER_HR"
echo "Net new high-rate      : $((AFTER_HR - BEFORE_HR))"

# ------------------------------------------------------------
# Identify files created during THIS run.
# ------------------------------------------------------------
find outputs \
    -type f \
    -name '*_highrate_truth.csv' \
    -newer "$RUNROOT/start.marker" \
    -print \
    2>/dev/null \
    | sort \
    > "$RUNROOT/highrate_files.txt"

NEW_COUNT="$(wc -l < "$RUNROOT/highrate_files.txt")"

echo "Files belonging to this run: $NEW_COUNT"

echo
echo "================================================================================"
echo "POST-RUN HIGH-RATE CSV QC"
echo "================================================================================"

RUNROOT="$RUNROOT" ./.venv/bin/python - <<'PY'
from pathlib import Path
import csv
import os
import numpy as np

runroot = Path(os.environ["RUNROOT"])
list_file = runroot / "highrate_files.txt"

paths = []

if list_file.exists():
    paths = [
        Path(x.strip())
        for x in list_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
        if x.strip()
    ]

required = {
    "timestamp",
    "accel_true_x",
    "accel_true_y",
    "accel_true_z",
    "gyro_true_x",
    "gyro_true_y",
    "gyro_true_z",
}

results = []

for p in paths:

    row = {
        "file": str(p),
        "rows": 0,
        "hz": np.nan,
        "finite": False,
        "status": "FAIL",
        "error": "",
    }

    try:
        with p.open(
            "r",
            encoding="utf-8",
            errors="strict",
            newline="",
        ) as f:
            lines = [
                line
                for line in f
                if not line.startswith("#")
            ]

        reader = csv.DictReader(lines)
        records = list(reader)

        fields = set(reader.fieldnames or [])

        if not required.issubset(fields):
            missing = sorted(required - fields)
            raise RuntimeError(
                f"missing columns: {missing}"
            )

        t = np.asarray(
            [float(r["timestamp"]) for r in records],
            dtype=float,
        )

        acc = np.asarray([
            [
                float(r["accel_true_x"]),
                float(r["accel_true_y"]),
                float(r["accel_true_z"]),
            ]
            for r in records
        ])

        gyro = np.asarray([
            [
                float(r["gyro_true_x"]),
                float(r["gyro_true_y"]),
                float(r["gyro_true_z"]),
            ]
            for r in records
        ])

        if len(t) < 2:
            raise RuntimeError("fewer than 2 samples")

        hz = float(
            1.0 / np.median(np.diff(t))
        )

        finite = bool(
            np.all(np.isfinite(t))
            and np.all(np.isfinite(acc))
            and np.all(np.isfinite(gyro))
        )

        if not finite:
            raise RuntimeError(
                "non-finite sensor values"
            )

        if not np.isclose(
            hz,
            100.0,
            atol=0.05,
        ):
            raise RuntimeError(
                f"unexpected output rate {hz:.6f} Hz"
            )

        row.update({
            "rows": len(t),
            "hz": hz,
            "finite": True,
            "status": "OK",
        })

    except Exception as exc:
        row["error"] = str(exc)

    results.append(row)

qc_csv = runroot / "highrate_qc.csv"

with qc_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "rows",
            "hz",
            "finite",
            "status",
            "error",
        ],
    )

    writer.writeheader()
    writer.writerows(results)

ok = sum(
    r["status"] == "OK"
    for r in results
)

bad = len(results) - ok

print("files checked :", len(results))
print("QC OK         :", ok)
print("QC failed     :", bad)
print("QC report     :", qc_csv)

(runroot / "summary.txt").write_text(
    "\n".join([
        f"highrate_files={len(results)}",
        f"qc_ok={ok}",
        f"qc_failed={bad}",
    ]) + "\n",
    encoding="utf-8",
)

if bad:
    print("\nFailed files:")
    for r in results:
        if r["status"] != "OK":
            print(" -", r["file"], ":", r["error"])
PY

QC_RC=$?

echo
echo "================================================================================"
echo "OVERNIGHT RUN SUMMARY"
echo "================================================================================"

echo "Batch exit code : $BATCH_RC"
echo "QC exit code    : $QC_RC"
echo "Run directory   : $RUNROOT"
echo "Log             : $LOG"
echo "Completed       : $(date -Is)"

echo "$BATCH_RC" > "$RUNROOT/batch_exit_code.txt"

if [ "$BATCH_RC" -eq 0 ]; then
    touch "$RUNROOT/BATCH_COMPLETE"
    echo
    echo "BATCH_COMPLETE"
else
    touch "$RUNROOT/BATCH_FAILED"
    echo
    echo "BATCH_FAILED - inspect batch.log tomorrow"
fi

exit "$BATCH_RC"
