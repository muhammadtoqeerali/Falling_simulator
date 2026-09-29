#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
SCRIPT="$ROOT/run_extended_35_profile_campaign.py"
CAMPAIGN="campaign_highrate_truth_v2_extended35"

cd "$ROOT"

LOGDIR="outputs/_highrate_overnight/$CAMPAIGN/logs"
mkdir -p "$LOGDIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/overnight_console_${STAMP}.log"
PIDFILE="outputs/_highrate_overnight/$CAMPAIGN/overnight.pid"
LATEST="outputs/_highrate_overnight/$CAMPAIGN/latest_log.txt"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "Starting extended 35-profile campaign at $(date)"
echo "Log: $LOG"

nohup nice -n 5 ./.venv/bin/python -u "$SCRIPT" \
    --root "$ROOT" \
    --campaign-name "$CAMPAIGN" \
    >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"
echo "$LOG" > "$LATEST"

echo "PID: $PID"
echo
echo "Monitor with:"
echo "  tail -f $ROOT/$LOG"
echo
echo "Process check:"
echo "  ps -fp $PID"
echo
echo "Progress table:"
echo "  $ROOT/outputs/_highrate_overnight/$CAMPAIGN/campaign_progress.csv"
echo
echo "NOTE: live_status.json appears only after the first run finishes."
