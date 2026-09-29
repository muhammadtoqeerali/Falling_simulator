#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project
PY="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/.venv/bin/python"

echo "======================================================================"
echo "TASK39 REPLAY MISMATCH DIAGNOSTIC V2 — NO RERUN"
echo "======================================================================"

"$PY" -m py_compile diagnose_task39_replay_mismatch_v2.py
echo "COMPILE: PASS"

rm -rf outputs/task39_replay_mismatch_diagnostic_v2
rm -f outputs/task39_replay_mismatch_diagnostic_v2.zip
rm -f outputs/task39_replay_mismatch_diagnostic_v2.zip.sha256

"$PY" -u diagnose_task39_replay_mismatch_v2.py \
  --output-dir outputs/task39_replay_mismatch_diagnostic_v2 \
  --zip-path outputs/task39_replay_mismatch_diagnostic_v2.zip \
  | tee outputs/task39_replay_mismatch_diagnostic_v2_run.log

echo
echo "FINAL FILES:"
ls -lh \
  outputs/task39_replay_mismatch_diagnostic_v2.zip \
  outputs/task39_replay_mismatch_diagnostic_v2.zip.sha256
