#!/usr/bin/env bash
set -euo pipefail

cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"

echo "======================================================================"
echo "HEIGHT-FALL PAPER EVIDENCE EXTRACTOR V5 — STANDALONE"
echo "======================================================================"

"$PY" -m py_compile prepare_height_fall_paper_evidence_v5.py
echo "COMPILE: PASS"

rm -rf outputs/paper_task39_height_fall_evidence_v5
rm -f outputs/paper_task39_height_fall_evidence_v5.zip
rm -f outputs/paper_task39_height_fall_evidence_v5.zip.sha256

"$PY" -u prepare_height_fall_paper_evidence_v5.py \
  --task-id 39 \
  --profile-id AUTO \
  --top-n 22 \
  --output-dir outputs/paper_task39_height_fall_evidence_v5 \
  --zip-path outputs/paper_task39_height_fall_evidence_v5.zip \
  | tee outputs/paper_task39_height_fall_evidence_v5_run.log

echo
echo "FINAL FILES:"
ls -lh \
  outputs/paper_task39_height_fall_evidence_v5.zip \
  outputs/paper_task39_height_fall_evidence_v5.zip.sha256
