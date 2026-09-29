#!/usr/bin/env bash
set -euo pipefail

ROOT="${MUJOCO_PROJECT_ROOT:-/mnt/hdd16T/ToqeerHomeBackup/mujoco_project}"
PY="$ROOT/.venv/bin/python"

cd "$ROOT"

echo "===================================================================================================="
echo "TASK39 CORRECTED396 RUNTIME-RESIDUE FORENSICS V6 — NO SIMULATOR RERUN"
echo "===================================================================================================="

if [[ ! -x "$PY" ]]; then
  echo "ERROR: expected project Python not executable: $PY" >&2
  exit 2
fi

"$PY" -m py_compile diagnose_task39_corrected396_runtime_residue_v6.py
echo "COMPILE: PASS"

# Hard safety check: diagnostic itself must not import simulator/runtime stack.
if grep -Eq '^[[:space:]]*(from|import)[[:space:]]+(mujoco|torch|humenv|metamotivo|scenario39)' \
    diagnose_task39_corrected396_runtime_residue_v6.py; then
  echo "ERROR: forbidden simulator/runtime import detected." >&2
  exit 3
fi

"$PY" diagnose_task39_corrected396_runtime_residue_v6.py

echo
echo "FINAL FILES:"
ls -lh \
  outputs/task39_corrected396_runtime_residue_forensics_v6.zip \
  outputs/task39_corrected396_runtime_residue_forensics_v6.zip.sha256
