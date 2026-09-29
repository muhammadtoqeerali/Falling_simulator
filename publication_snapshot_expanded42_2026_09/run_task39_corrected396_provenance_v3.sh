#!/usr/bin/env bash
set -euo pipefail

ROOT="${MUJOCO_PROJECT_ROOT:-/mnt/hdd16T/ToqeerHomeBackup/mujoco_project}"
PY="$ROOT/.venv/bin/python"

cd "$ROOT"

echo "================================================================================"
echo "TASK39 CORRECTED396 PROVENANCE RECONCILIATION V3 — NO SIMULATOR RERUN"
echo "================================================================================"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: expected project Python not executable: $PY" >&2
  exit 2
fi

"$PY" -m py_compile diagnose_task39_corrected396_provenance_v3.py
echo "COMPILE: PASS"

# Explicit safety checks: the diagnostic must not import/run simulator modules.
if grep -Eq '^[[:space:]]*(from|import)[[:space:]]+(mujoco|torch|humenv|metamotivo|scenario39)' diagnose_task39_corrected396_provenance_v3.py; then
  echo "ERROR: safety check found forbidden runtime import." >&2
  exit 3
fi

"$PY" diagnose_task39_corrected396_provenance_v3.py

echo
echo "FINAL FILES:"
ls -lh \
  outputs/task39_corrected396_provenance_v3.zip \
  outputs/task39_corrected396_provenance_v3.zip.sha256
