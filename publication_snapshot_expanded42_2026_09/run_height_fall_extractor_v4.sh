#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"

echo "======================================================================"
echo "HEIGHT-FALL EXTRACTOR V4 — ACTUAL-FUNCTION PATCH"
echo "======================================================================"

"$PY" -u patch_height_fall_extractor_v4.py

echo
echo "======================================================================"
echo "RERUN TASK 39 EXTRACTION"
echo "======================================================================"

rm -rf outputs/paper_task39_height_fall_evidence_v4
rm -f outputs/paper_task39_height_fall_evidence_v4.zip
rm -f outputs/paper_task39_height_fall_evidence_v4.zip.sha256

"$PY" -u prepare_height_fall_paper_evidence.py \
  --task-id 39 \
  --profile-id AUTO \
  --top-n 22 \
  --output-dir outputs/paper_task39_height_fall_evidence_v4 \
  --zip-path outputs/paper_task39_height_fall_evidence_v4.zip \
  | tee outputs/paper_task39_height_fall_evidence_v4_run.log

echo
echo "======================================================================"
echo "FINAL FILES"
echo "======================================================================"

ls -lh \
  outputs/paper_task39_height_fall_evidence_v4.zip \
  outputs/paper_task39_height_fall_evidence_v4.zip.sha256
