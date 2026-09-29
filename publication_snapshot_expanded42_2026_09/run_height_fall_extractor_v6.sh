#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project
PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"

echo "======================================================================"
echo "HEIGHT-FALL PAPER EVIDENCE EXTRACTOR V6 — ROBUST"
echo "======================================================================"

"$PY" -m py_compile prepare_height_fall_paper_evidence_v6.py
echo "COMPILE: PASS"

rm -rf outputs/paper_task39_height_fall_evidence_v6
rm -f outputs/paper_task39_height_fall_evidence_v6.zip
rm -f outputs/paper_task39_height_fall_evidence_v6.zip.sha256

"$PY" -u prepare_height_fall_paper_evidence_v6.py \
  --task-id 39 \
  --profile-id AUTO \
  --output-dir outputs/paper_task39_height_fall_evidence_v6 \
  --zip-path outputs/paper_task39_height_fall_evidence_v6.zip \
  | tee outputs/paper_task39_height_fall_evidence_v6_run.log

echo
echo "FINAL FILES:"
ls -lh \
  outputs/paper_task39_height_fall_evidence_v6.zip \
  outputs/paper_task39_height_fall_evidence_v6.zip.sha256
