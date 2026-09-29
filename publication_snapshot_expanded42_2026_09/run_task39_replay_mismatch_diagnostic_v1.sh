#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project
PY="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/.venv/bin/python"

echo "======================================================================"
echo "TASK39 REPLAY MISMATCH DIAGNOSTIC — NO RERUN"
echo "======================================================================"

"$PY" -m py_compile diagnose_task39_replay_mismatch_v1.py
echo "COMPILE: PASS"

rm -rf outputs/task39_replay_mismatch_diagnostic_v1
rm -f outputs/task39_replay_mismatch_diagnostic_v1.zip
rm -f outputs/task39_replay_mismatch_diagnostic_v1.zip.sha256

"$PY" -u diagnose_task39_replay_mismatch_v1.py \
  --output-dir outputs/task39_replay_mismatch_diagnostic_v1 \
  --zip-path outputs/task39_replay_mismatch_diagnostic_v1.zip \
  | tee outputs/task39_replay_mismatch_diagnostic_v1_run.log

echo
echo "FINAL FILES:"
ls -lh \
  outputs/task39_replay_mismatch_diagnostic_v1.zip \
  outputs/task39_replay_mismatch_diagnostic_v1.zip.sha256
