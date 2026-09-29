#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

PY="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/.venv/bin/python"

# No X server is required. EGL is requested only for optional offscreen RGB
# phase renders. The physics/state/contact collection works without those PNGs.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset DISPLAY || true

echo "======================================================================"
echo "TASK 39 / P020 EXACT POSE REPLAY V2 — HEADLESS"
echo "======================================================================"

"$PY" -m py_compile task39_p020_exact_pose_replay_v2_headless.py
echo "COMPILE: PASS"

rm -rf outputs/task39_p020_exact_pose_replay_v2_headless
rm -f outputs/task39_p020_exact_pose_replay_v2_headless.zip
rm -f outputs/task39_p020_exact_pose_replay_v2_headless.zip.sha256

"$PY" -u task39_p020_exact_pose_replay_v2_headless.py \
  --output-dir outputs/task39_p020_exact_pose_replay_v2_headless \
  --zip-path outputs/task39_p020_exact_pose_replay_v2_headless.zip \
  | tee outputs/task39_p020_exact_pose_replay_v2_headless_run.log

echo
echo "FINAL FILES:"
ls -lh \
  outputs/task39_p020_exact_pose_replay_v2_headless.zip \
  outputs/task39_p020_exact_pose_replay_v2_headless.zip.sha256

echo
echo "PAPER GATE:"
cat outputs/task39_p020_exact_pose_replay_v2_headless/USE_FOR_PAPER.json
