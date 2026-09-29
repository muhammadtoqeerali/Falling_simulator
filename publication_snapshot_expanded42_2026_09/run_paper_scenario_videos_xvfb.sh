#!/usr/bin/env bash

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project || exit 1

PY="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/.venv/bin/python"

mkdir -p outputs/PAPER_SCENARIO_VIDEO_REGEN

echo "============================================================"
echo "PAPER SCENARIO VIDEO REGENERATION"
echo "============================================================"
echo "Tasks: 23,25,39,43"
echo "23 = chair/getting-up fall"
echo "25 = seated/fainting fall"
echo "39 = forward fall from height"
echo "43 = forward fall climbing ladder"
echo "Started: $(date)"
echo "============================================================"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export LIBGL_ALWAYS_SOFTWARE=1

exec /usr/bin/xvfb-run \
    -a \
    -s "-screen 0 1920x1080x24 -ac +extension GLX +render -noreset" \
    "$PY" -u batch_run_all_labeled_video.py \
        --age 25 \
        --height 1.62 \
        --sex female \
        --weight 58 \
        --tasks "23,25,39,43" \
        --video-fps 30 \
        --video-size 1920x1008 \
        --continue-on-error
