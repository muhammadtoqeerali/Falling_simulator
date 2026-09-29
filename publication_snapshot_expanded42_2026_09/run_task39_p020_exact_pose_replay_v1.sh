#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

PY="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/.venv/bin/python"

echo "======================================================================"
echo "TASK 39 / P020 EXACT POSE REPLAY"
echo "======================================================================"

"$PY" -m py_compile task39_p020_exact_pose_replay_v1.py
echo "COMPILE: PASS"

rm -rf outputs/task39_p020_exact_pose_replay_v1
rm -f outputs/task39_p020_exact_pose_replay_v1.zip
rm -f outputs/task39_p020_exact_pose_replay_v1.zip.sha256

"$PY" -u task39_p020_exact_pose_replay_v1.py \
  --output-dir outputs/task39_p020_exact_pose_replay_v1 \
  --zip-path outputs/task39_p020_exact_pose_replay_v1.zip \
  | tee outputs/task39_p020_exact_pose_replay_v1_run.log

echo
echo "FINAL FILES:"
ls -lh \
  outputs/task39_p020_exact_pose_replay_v1.zip \
  outputs/task39_p020_exact_pose_replay_v1.zip.sha256

echo
echo "PAPER GATE:"
cat outputs/task39_p020_exact_pose_replay_v1/USE_FOR_PAPER.json
