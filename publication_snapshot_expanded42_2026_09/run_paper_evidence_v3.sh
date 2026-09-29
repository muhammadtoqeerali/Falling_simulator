#!/usr/bin/env bash
set -u

PROJECT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"

cd "$PROJECT" || exit 1

"$PY" -m py_compile curate_paper_evidence_v3.py || exit 2

"$PY" -u curate_paper_evidence_v3.py \
  --stage "$PROJECT/outputs/paper_evidence_archive_v1" \
  --zip "$PROJECT/outputs/paper_evidence_archive_v1_FULL_V3.zip"
