#!/usr/bin/env bash
set -euo pipefail

ROOT="${MUJOCO_PROJECT_ROOT:-/mnt/hdd16T/ToqeerHomeBackup/mujoco_project}"
PY="$ROOT/.venv/bin/python"

cd "$ROOT"

echo "================================================================================================"
echo "TASK39 CORRECTED396 HISTORICAL CALL-CHAIN TRACE V5 — NO SIMULATOR RERUN"
echo "================================================================================================"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: expected project Python not executable: $PY" >&2
  exit 2
fi

"$PY" -m py_compile trace_task39_corrected396_callchain_v5.py
echo "COMPILE: PASS"

if grep -Eq '^[[:space:]]*(from|import)[[:space:]]+(mujoco|torch|humenv|metamotivo|scenario39)' \
    trace_task39_corrected396_callchain_v5.py; then
  echo "ERROR: forbidden simulator/runtime import detected." >&2
  exit 3
fi

"$PY" trace_task39_corrected396_callchain_v5.py

echo
echo "FINAL FILES:"
ls -lh \
  outputs/task39_corrected396_callchain_trace_v5.zip \
  outputs/task39_corrected396_callchain_trace_v5.zip.sha256
