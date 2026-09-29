#!/usr/bin/env bash
set -euo pipefail

ROOT="${MUJOCO_PROJECT_ROOT:-/mnt/hdd16T/ToqeerHomeBackup/mujoco_project}"
PY="$ROOT/.venv/bin/python"

cd "$ROOT"

echo "=============================================================================================="
echo "TASK39 CORRECTED396 PROVENANCE ADJUDICATION V4 — NO SIMULATOR RERUN"
echo "=============================================================================================="

if [[ ! -x "$PY" ]]; then
  echo "ERROR: expected project Python not executable: $PY" >&2
  exit 2
fi

"$PY" -m py_compile adjudicate_task39_corrected396_provenance_v4.py
echo "COMPILE: PASS"

# Hard safety check: no imports of simulator/runtime ML stack.
if grep -Eq '^[[:space:]]*(from|import)[[:space:]]+(mujoco|torch|humenv|metamotivo|scenario39)' \
    adjudicate_task39_corrected396_provenance_v4.py; then
  echo "ERROR: forbidden simulator/runtime import detected." >&2
  exit 3
fi

"$PY" adjudicate_task39_corrected396_provenance_v4.py

echo
echo "FINAL FILES:"
ls -lh \
  outputs/task39_corrected396_provenance_adjudication_v4.zip \
  outputs/task39_corrected396_provenance_adjudication_v4.zip.sha256
