TASK39 P020 PRE-CAMPAIGN SOURCE EQUIVALENCE REPLAY V7
=====================================================

THIS IS THE FIRST AUTHORIZED RERUN AFTER V2-V6 REVIEW.

WHY WE ARE PROCEEDING NOW
-------------------------
The diagnostic sequence established:
- the September-1 standalone replay source is not equivalent to corrected396;
- the mismatch is source/runtime related, not a coordinate offset;
- the strongest genuine historical source is scenario39_legacy.py at:

    commit:
      db9a7b67163edcf58d64d66afd4ed3271b34899b

    SHA256:
      ae5180d9afdc1a76b3b7bcc82e8d3b6f4fb65c005688734f96bdfd583d81f399

- that historical source differs from the failed September-1 source;
- V4 found two pre-campaign controller references to it;
- later broad residue rankings were contaminated by self-generated diagnostics.

Rather than continue indefinite provenance searches, V7 performs ONE controlled
equivalence test using the historical source and the already-proven V2
headless/full-state/contact/marker replay instrumentation.

IMPORTANT SCIENTIFIC POINT
--------------------------
The final corrected high-rate pipeline calls:

    mj_rnePostConstraint()
    before
    mj_objectAcceleration()

This is required to obtain body acceleration (`cacc`) for the object
acceleration export. Canonical corrected396 CSV signals remain the primary
paper signal source. V7's decisive question is whether the historical source
reproduces the canonical trajectory closely enough to unlock exact whole-body
pose/state/contact evidence for the figure.

V7 DOES NOT:
- overwrite scenario39_legacy.py;
- loosen any threshold;
- replace canonical corrected396 signals;
- approve pose files unless ALL strict equivalence criteria pass.

STRICT ORIGINAL THRESHOLDS
--------------------------
pelvis-height RMSE       <= 0.03 m
sensor-position 3D RMSE <= 0.04 m
peak-impact time delta  <= 0.10 s
total-duration delta    <= 0.15 s

RUN EXACTLY
-----------
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o task39_p020_precampaign_equivalence_replay_v7_bundle.zip

chmod +x run_task39_p020_precampaign_equivalence_replay_v7.sh

bash run_task39_p020_precampaign_equivalence_replay_v7.sh

EXIT STATUS
-----------
0  = strict equivalence PASS
10 = replay completed but strict equivalence FAIL
5  = technical error

Do not interpret exit 10 as a shell failure; it means the scientific gate
correctly remained blocked.

UPLOAD AFTER RUN
----------------
Upload BOTH:

  outputs/task39_p020_precampaign_equivalence_replay_v7.zip
  outputs/task39_p020_precampaign_equivalence_replay_v7.zip.sha256

and paste the terminal V7 FINAL DECISION.

IF STRICT EQUIVALENCE PASSES
----------------------------
The ZIP will contain:

  05_HISTORICAL_EQUIVALENCE_DECISION/
      HISTORICAL_SOURCE_REPLAY_DECISION.json
      FIGURE_EVIDENCE_READY.json

At that point the validated replay pose/marker/joint/contact/full-state evidence
can be combined with the canonical corrected396 kinematics/impact/IMU signals
to build the final publication-quality Task-39 figure at the seven locked phase
timestamps.
