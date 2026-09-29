#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project
PY=./.venv/bin/python
RUNNER=run_phase2_lightweight_paper_final_fold.py
OUT=outputs/phase2_lightweight_paper_final_campaign_v1
GPU="${CAMPAIGN_GPU:-1}"
mkdir -p "$OUT"
EXPS=(
  EXP01_REAL_ONLY
  EXP02_SIM_FALL_SUBSTITUTION
  EXP03_MIX20
  EXP04_MIX50
  EXP05_MIX70
  EXP06_MIX100
)
for EXP in "${EXPS[@]}"; do
  echo "===================================================================================================="
  echo "STARTING COMPLETE EXPERIMENT: $EXP"
  echo "===================================================================================================="
  for FOLD in 0 1 2 3 4; do
    COMPLETE="$OUT/$EXP/fold_$FOLD/COMPLETE"
    if [[ -f "$COMPLETE" ]]; then
      echo "$EXP | FOLD $FOLD ALREADY COMPLETE -- SKIP"
      continue
    fi
    echo "$EXP | STARTING FOLD $FOLD"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$RUNNER" --experiment "$EXP" --fold "$FOLD" --output-root "$OUT"
    test -f "$COMPLETE"
    echo "$EXP | FOLD $FOLD COMPLETE"
  done
  echo "ALL 5 FOLDS COMPLETE: $EXP"
done
"$PY" -u aggregate_phase2_lightweight_paper_final.py
echo "FINAL LIGHTWEIGHT-PAPER CAMPAIGN COMPLETE"
