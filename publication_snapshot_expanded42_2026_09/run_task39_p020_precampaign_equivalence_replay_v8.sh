#!/usr/bin/env bash
set -uo pipefail

ROOT="${MUJOCO_PROJECT_ROOT:-/mnt/hdd16T/ToqeerHomeBackup/mujoco_project}"
PY="$ROOT/.venv/bin/python"

cd "$ROOT"

echo "========================================================================================================"
echo "TASK39 P020 PRE-CAMPAIGN SOURCE EQUIVALENCE REPLAY V8 — RESOLVER FIX"
echo "========================================================================================================"

"$PY" -m py_compile run_task39_p020_precampaign_equivalence_replay_v8.py
echo "COMPILE: PASS"

echo "V7 technical bug fixed: V8 cannot select V7/V8 as the replay driver."
echo "Historical source commit: db9a7b67163edcf58d64d66afd4ed3271b34899b"
echo "Expected historical source SHA256: ae5180d9afdc1a76b3b7bcc82e8d3b6f4fb65c005688734f96bdfd583d81f399"
echo

"$PY" run_task39_p020_precampaign_equivalence_replay_v8.py
RC=$?

echo
echo "RUN EXIT STATUS: $RC"
echo "  0  = strict equivalence PASS"
echo " 10  = controlled replay completed but strict equivalence FAIL"
echo "  5  = replay/gate technical error"
echo

echo "FINAL FILES (if generated):"
ls -lh \
  outputs/task39_p020_precampaign_equivalence_replay_v8.zip \
  outputs/task39_p020_precampaign_equivalence_replay_v8.zip.sha256 2>/dev/null || true

exit "$RC"
