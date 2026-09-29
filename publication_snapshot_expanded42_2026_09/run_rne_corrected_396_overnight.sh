#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

ROOT="outputs/_highrate_overnight/campaign_highrate_truth_v3_rne_corrected396"
mkdir -p "$ROOT/logs"

TS=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/logs/overnight_console_${TS}.log"

printf '%s\n' "$LOG" > "$ROOT/latest_log.txt"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export LIBGL_ALWAYS_SOFTWARE=1
export MUJOCO_DISABLE_NATIVE_VIDEO=1

echo "============================================================"
echo "RNE-CORRECTED SYNTHETIC CAMPAIGN"
echo "============================================================"
echo "Expected profiles : 22"
echo "Expected tasks    : 18"
echo "Expected runs     : 396"
echo "Native video      : DISABLED"
echo "Retry failures    : ENABLED"
echo "Runner            : run_rne_corrected_396_campaign.py"
echo "Log               : $LOG"
echo "============================================================"

nice -n 5 \
./.venv/bin/python \
-u run_rne_corrected_396_campaign.py \
--retry-failed \
2>&1 | tee "$LOG"
