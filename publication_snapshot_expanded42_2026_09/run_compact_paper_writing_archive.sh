#!/usr/bin/env bash
set -u
PROJECT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"
cd "$PROJECT" || exit 1
"$PY" -m py_compile build_compact_paper_writing_archive.py || exit 2
"$PY" -u build_compact_paper_writing_archive.py \
  --output-dir "$PROJECT/outputs/paper_writing_evidence_COMPACT" \
  --zip "$PROJECT/outputs/paper_writing_evidence_COMPACT.zip"
