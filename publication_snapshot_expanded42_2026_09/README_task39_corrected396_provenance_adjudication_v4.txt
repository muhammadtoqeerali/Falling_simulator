TASK39 CORRECTED396 PROVENANCE ADJUDICATION V4
==============================================

WHY THIS FOLLOW-UP IS REQUIRED
------------------------------
V3 completed safely and correctly preserved the BLOCK gate, but its terminal
ranking exposed a provenance false positive:

  Top static candidate:
    diagnose_task39_replay_mismatch_v1.py

That file is diagnostic code. It contains canonical/replay strings because it
was written to compare them; it cannot be accepted as the historical simulator
runtime that generated corrected396.

V4 fixes that specific methodological problem.

V4 SAFETY
---------
V4 is STATIC / READ-ONLY.

It does NOT:
- import scenario39;
- import MuJoCo;
- import HumEnv / Meta Motivo / torch;
- execute a scenario;
- rerun Task 39;
- modify simulator source;
- change canonical data;
- loosen replay-equivalence thresholds;
- unblock the paper.

WHAT V4 DOES
------------
1. Restricts runtime candidates to the scenario39 source family and backups.
2. Excludes diagnostic/replay/audit/extractor/patch/test/provenance code.
3. Finds scenario39 paths and versions preserved in Git BEFORE the August 26,
   2026 corrected396 campaign.
4. Saves the newest pre-campaign historical source blobs as TEXT ONLY for review.
5. Searches likely campaign/runner code for Task-39/scenario39 call-chain lines.
6. Compares:
     - canonical `fall_scenario39...` export signature,
     - failed-replay `fall_forward_height...` signature,
     - run_with_subject,
     - mj_rnePostConstraint,
     - Task39HeightLayer/controller instrumentation,
     - static frame/step timing hints.
7. Produces an adjudicated ranking and proof grade.
8. Keeps the paper gate BLOCKED.

RUN EXACTLY
-----------
Place the bundle in:

  /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

Then:

  cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

  unzip -o task39_corrected396_provenance_adjudication_v4_bundle.zip

  chmod +x run_task39_corrected396_provenance_adjudication_v4.sh

  bash run_task39_corrected396_provenance_adjudication_v4.sh

EXPECTED EARLY OUTPUT
---------------------
  COMPILE: PASS
  Paper gate: BLOCK_REMAINS
  Simulator rerun performed: NO
  Replay authorized now: NO

UPLOAD AFTER RUN
----------------
Upload BOTH:

  outputs/task39_corrected396_provenance_adjudication_v4.zip
  outputs/task39_corrected396_provenance_adjudication_v4.zip.sha256

Also paste the terminal summary.

DO NOT RERUN TASK 39 YET.
