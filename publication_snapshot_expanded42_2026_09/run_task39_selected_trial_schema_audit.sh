#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"

"$PY" -m py_compile audit_height_fall_selected_trial_schema_v1.py
echo "COMPILE: PASS"

rm -rf outputs/task39_selected_trial_schema_audit_v1
rm -f outputs/task39_selected_trial_schema_audit_v1.zip
rm -f outputs/task39_selected_trial_schema_audit_v1.zip.sha256

"$PY" -u audit_height_fall_selected_trial_schema_v1.py \
  --v5-dir outputs/paper_task39_height_fall_evidence_v5 \
  --search-root outputs/_highrate_overnight \
  --output-dir outputs/task39_selected_trial_schema_audit_v1 \
  --zip-path outputs/task39_selected_trial_schema_audit_v1.zip \
  | tee outputs/task39_selected_trial_schema_audit_v1_run.log

echo
echo "FINAL FILES:"
ls -lh \
  outputs/task39_selected_trial_schema_audit_v1.zip \
  outputs/task39_selected_trial_schema_audit_v1.zip.sha256
