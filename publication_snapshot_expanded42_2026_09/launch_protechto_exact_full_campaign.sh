#!/usr/bin/env bash

PROJECT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"
OUT="$PROJECT/outputs/protechto_exact_full_campaign_v1"

cd "$PROJECT" || exit 1
mkdir -p "$OUT"

"$PY" -u preflight_protechto_exact_full_campaign.py
RC=$?
if [ "$RC" -ne 0 ]; then
    echo
    echo "FULL CAMPAIGN PREFLIGHT FAILED -- NOTHING STARTED"
    exit "$RC"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/full_campaign_${STAMP}.log"

nohup "$PY" -u run_protechto_exact_full_campaign.py \
    --output-root "$OUT" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!

echo "$PID" > "$OUT/LATEST_PID"
echo "$LOG" > "$OUT/LATEST_LOG"

disown

echo
echo "======================================================================"
echo "PROTECHTO EXACT FULL CAMPAIGN STARTED"
echo "======================================================================"
echo "PID : $PID"
echo "LOG : $LOG"
echo "GPU : 1"
echo "Conditions:"
echo "  EXP01_REAL_ONLY"
echo "  EXP02_SIM_FALL_SUBSTITUTION"
echo "  EXP03_SIM_ONLY_FULL"
echo "  EXP04_MIX20"
echo "  EXP05_MIX50"
echo "  EXP06_MIX70"
echo "  EXP07_MIX100"
echo "======================================================================"
