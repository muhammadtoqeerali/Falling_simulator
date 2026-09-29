#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
SCRIPT="$ROOT/run_extended_35_profile_campaign.py"
CAMPAIGN="campaign_highrate_truth_v2_extended35"

cd "$ROOT"

mkdir -p "outputs/_highrate_overnight/$CAMPAIGN/logs"

LOG="outputs/_highrate_overnight/$CAMPAIGN/logs/overnight_console.log"
PIDFILE="outputs/_highrate_overnight/$CAMPAIGN/overnight.pid"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "Starting extended 35-profile campaign at $(date)"
echo "Log: $LOG"

nohup nice -n 5 ./.venv/bin/python -u "$SCRIPT" \
    --root "$ROOT" \
    --campaign-name "$CAMPAIGN" \
    > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"

echo "PID: $PID"
echo
echo "Monitor with:"
echo "  tail -f $ROOT/$LOG"
echo
echo "Check status with:"
echo "  cat $ROOT/outputs/_highrate_overnight/$CAMPAIGN/live_status.json"
echo
echo "Process check:"
echo "  ps -fp $PID"
