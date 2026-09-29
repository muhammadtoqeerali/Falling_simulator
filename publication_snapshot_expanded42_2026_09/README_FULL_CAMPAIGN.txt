PROTECHTO EXACT FULL CAMPAIGN

Purpose
-------
Run the complete seven-condition campaign once, sequentially on GPU 1, after one strict preflight.
The canonical physical Protechto source tree is never edited.

Canonical source:
  /mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master

Execution mirror:
  /mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/protechto_clean_execution_mirror_v1

Python environment:
  /mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python

Physical input:
  data/dataset/segments/300ms_50ov_npseg_filt_binary
  untouched; no manual 150-ms trimming.

Synthetic input:
  outputs/phase2_corrected_event_dataset_v1/synthetic_X.npy
  outputs/phase2_corrected_event_dataset_v1/synthetic_y.npy
  The cache preparer requires a row-aligned metadata CSV under this same source root.
  It enforces 20080 windows: 18824 Activity and 1256 Falling.
  It does no temporal trimming. It only converts m/s^2 -> mg and deg/s -> mdps.

Conditions
----------
EXP01_REAL_ONLY
  Original physical train/val/test K-fold pipeline.

EXP02_SIM_FALL_SUBSTITUTION
  Physical training Activity + simulated Falling.
  Synthetic Falling count is matched to the physical training Falling count.
  Physical validation/test are unchanged.

EXP03_SIM_ONLY_FULL
  Simulated pre-fall Activity + simulated Falling.
  Whole simulated trial IDs are split using 5-fold KFold(random_state=42), with 20% validation from the outer train IDs.
  Segment metrics are valid binary metrics. Event metrics are fully informative only if the simulated metadata contains both fall-event and activity-only trials; otherwise event specificity is a one-class caveat.

EXP04_MIX20
EXP05_MIX50
EXP06_MIX70
EXP07_MIX100
  Physical training set + synthetic Falling additions equal to 20/50/70/100% of the real training Falling count.
  Physical validation/test remain unchanged.

Synthetic draws are deterministic, trial-balanced, unique-first, and only reuse samples after the entire unique Falling pool is exhausted.

Training/evaluation
-------------------
Model, IMUNormalizer, Predictor, AdamW, CrossEntropyLoss, lr/wd, 100 epochs, patience 20, batch 64, and GPU selection all come from the clean Protechto implementation/configuration.

For physical-test conditions, final evaluation calls the untouched clean test.py, so the historical prediction_bias=0.65 and two-consecutive-window event threshold are used exactly.

Launch
------
  bash launch_protechto_exact_full_campaign.sh

Monitor
-------
  OUT="outputs/protechto_exact_full_campaign_v1"
  PID=$(cat "$OUT/LATEST_PID")
  LOG=$(cat "$OUT/LATEST_LOG")
  ps -fp "$PID"
  nvidia-smi -i 1
  tail -n 120 "$LOG"

Progress
--------
  grep -E 'START PHYSICAL|START SIMULATION|FOLD [1-5] COMPLETE|CONDITION COMPLETE|CAMPAIGN COMPLETE|GATE: PASS' "$LOG" | tail -n 200

Final expected markers
----------------------
  outputs/protechto_exact_full_campaign_v1/FINAL_CAMPAIGN_COMPLETE
  PROTECHTO EXACT FULL CAMPAIGN ARTIFACT GATE: PASS
  PROTECHTO EXACT FULL CAMPAIGN COMPLETE
