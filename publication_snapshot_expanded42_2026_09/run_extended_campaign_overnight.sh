#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
SCRIPT="$ROOT/run_extended_35_profile_campaign.py"
CAMPAIGN="campaign_highrate_truth_v2_extended35"

cd "$ROOT"

if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "ERROR: xvfb-run is not installed."
    echo "Install with: sudo apt-get install -y xvfb xauth"
    exit 2
fi

LOGDIR="outputs/_highrate_overnight/$CAMPAIGN/logs"
mkdir -p "$LOGDIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/overnight_console_${STAMP}.log"
PIDFILE="outputs/_highrate_overnight/$CAMPAIGN/overnight.pid"
LATEST="outputs/_highrate_overnight/$CAMPAIGN/latest_log.txt"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export LIBGL_ALWAYS_SOFTWARE=1

echo "Starting extended 35-profile campaign at $(date)"
echo "Virtual display: xvfb-run"
echo "Log: $LOG"

nohup nice -n 5 xvfb-run -a \
    -s "-screen 0 1280x1024x24 -nolisten tcp" \
    ./.venv/bin/python -u "$SCRIPT" \
    --root "$ROOT" \
    --campaign-name "$CAMPAIGN" \
    >> "$LOG" 2>&1 &

PID=$!

echo "$PID" > "$PIDFILE"
echo "$LOG" > "$LATEST"

sleep 3

if ps -p "$PID" >/dev/null 2>&1; then
    echo "PID: $PID"
    echo
    echo "Monitor:"
    echo "  tail -f $ROOT/$LOG"
    echo
    echo "Live status:"
    echo "  cat $ROOT/outputs/_highrate_overnight/$CAMPAIGN/live_status.json"
else
    echo "Campaign exited during startup."
    echo "Inspect:"
    echo "  tail -150 $ROOT/$LOG"
    exit 1
fi
