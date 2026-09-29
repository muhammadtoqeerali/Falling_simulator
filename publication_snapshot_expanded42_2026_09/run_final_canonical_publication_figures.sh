#!/usr/bin/env bash
set -u

PROJECT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"

cd "$PROJECT" || exit 1

"$PY" -m py_compile build_final_canonical_publication_figures.py || exit 2

"$PY" -u build_final_canonical_publication_figures.py \
  --stage "$PROJECT/outputs/paper_evidence_archive_v1" \
  --zip "$PROJECT/outputs/paper_evidence_archive_FINAL.zip"
