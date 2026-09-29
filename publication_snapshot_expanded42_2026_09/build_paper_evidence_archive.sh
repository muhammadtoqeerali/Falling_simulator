#!/usr/bin/env bash
set -u

PROJECT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"

cd "$PROJECT" || exit 1

"$PY" -m py_compile build_paper_evidence_archive.py || exit 2

"$PY" -u build_paper_evidence_archive.py \
  --output-dir "$PROJECT/outputs/paper_evidence_archive_v1" \
  --zip-path "$PROJECT/outputs/paper_evidence_archive_v1.zip"
