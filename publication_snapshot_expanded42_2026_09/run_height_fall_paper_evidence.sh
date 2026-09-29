#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project
PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"
"$PY" -m py_compile prepare_height_fall_paper_evidence.py
echo "COMPILE: PASS"
"$PY" -u prepare_height_fall_paper_evidence.py \
  --task-id 39 \
  --profile-id AUTO \
  --top-n 8 \
  --output-dir outputs/paper_task39_height_fall_evidence_v1 \
  --zip-path outputs/paper_task39_height_fall_evidence_v1.zip \
  | tee outputs/paper_task39_height_fall_evidence_v1_run.log

echo
echo "FINAL FILES:"
ls -lh outputs/paper_task39_height_fall_evidence_v1.zip outputs/paper_task39_height_fall_evidence_v1.zip.sha256
