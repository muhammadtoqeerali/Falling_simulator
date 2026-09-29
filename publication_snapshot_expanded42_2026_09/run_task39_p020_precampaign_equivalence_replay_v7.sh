#!/usr/bin/env bash
set -uo pipefail

ROOT="${MUJOCO_PROJECT_ROOT:-/mnt/hdd16T/ToqeerHomeBackup/mujoco_project}"
PY="$ROOT/.venv/bin/python"

cd "$ROOT"

echo "========================================================================================================"
echo "TASK39 P020 PRE-CAMPAIGN SOURCE EQUIVALENCE REPLAY V7 — ONE CONTROLLED REPLAY"
echo "========================================================================================================"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: expected project Python not executable: $PY" >&2
  exit 2
fi

"$PY" -m py_compile run_task39_p020_precampaign_equivalence_replay_v7.py
echo "COMPILE: PASS"

echo "Historical source commit: db9a7b67163edcf58d64d66afd4ed3271b34899b"
echo "Expected historical source SHA256: ae5180d9afdc1a76b3b7bcc82e8d3b6f4fb65c005688734f96bdfd583d81f399"
echo "This run will preserve and restore the previous failed V2 replay."
echo "The project scenario39_legacy.py file will NOT be replaced on disk."
echo

"$PY" run_task39_p020_precampaign_equivalence_replay_v7.py
RC=$?

echo
echo "RUN EXIT STATUS: $RC"
echo "  0  = strict equivalence PASS"
echo " 10  = controlled replay completed but strict equivalence FAIL"
echo "  5  = replay/gate technical error"
echo

echo "FINAL FILES (if generated):"
ls -lh \
  outputs/task39_p020_precampaign_equivalence_replay_v7.zip \
  outputs/task39_p020_precampaign_equivalence_replay_v7.zip.sha256 2>/dev/null || true

# Do not hide the scientific gate status.
exit "$RC"
