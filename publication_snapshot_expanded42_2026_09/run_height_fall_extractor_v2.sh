#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"

echo "======================================================================"
echo "PATCH HEIGHT-FALL EXTRACTOR PROFILE PARSER"
echo "======================================================================"
"$PY" -u patch_height_fall_extractor_v2.py

echo
"$PY" -m py_compile prepare_height_fall_paper_evidence.py
echo "COMPILE AFTER PATCH: PASS"

echo
echo "======================================================================"
echo "RERUN TASK 39 EXTRACTION"
echo "======================================================================"

"$PY" -u prepare_height_fall_paper_evidence.py \
  --task-id 39 \
  --profile-id AUTO \
  --top-n 12 \
  --output-dir outputs/paper_task39_height_fall_evidence_v2 \
  --zip-path outputs/paper_task39_height_fall_evidence_v2.zip \
  | tee outputs/paper_task39_height_fall_evidence_v2_run.log

echo
echo "======================================================================"
echo "FINAL FILES"
echo "======================================================================"
ls -lh \
  outputs/paper_task39_height_fall_evidence_v2.zip \
  outputs/paper_task39_height_fall_evidence_v2.zip.sha256
