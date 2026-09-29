TASK39 CORRECTED396 HISTORICAL CALL-CHAIN TRACE V5
==================================================

V4 RESULT BEING FOLLOWED
------------------------
V4 correctly excluded diagnostic false positives and found:

  Top true candidate:
    GIT_PRECAMPAIGN scenario39_legacy.py

  commit:
    db9a7b67163edcf58d64d66afd4ed3271b34899b

  SHA256:
    ae5180d9afdc1a76b3b7bcc82e8d3b6f4fb65c005688734f96bdfd583d81f399

  same as failed replay source:
    NO

  pre-campaign controller references:
    2

But V4 remained UNRESOLVED because the historical source contains the raw
`fall_forward_height` signature while corrected396 stores the canonical file as
`fall_scenario39...`.

V5 does NOT treat that filename difference as proof of a different source.
A campaign wrapper may have normalized or renamed outputs.

V5 asks exactly:
1. Which controller files generated the two pre-campaign source references?
2. What did those controllers look like in Git before the August 26 campaign?
3. Did they map Task 39 to scenario39_legacy?
4. Did they copy/move/rename/package raw scenario output?
5. Did they generate or refer to `fall_scenario39...` independently of the raw
   scenario source?
6. Does this explain the filename contradiction?

SAFETY
------
V5 is STATIC / READ-ONLY.
It does not import or run scenario39, MuJoCo, HumEnv, Meta Motivo, or torch.
It does not modify source or canonical data.
It does not authorize replay.
Paper gate remains BLOCKED.

RUN EXACTLY
-----------
  cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

  unzip -o task39_corrected396_callchain_trace_v5_bundle.zip

  chmod +x run_task39_corrected396_callchain_trace_v5.sh

  bash run_task39_corrected396_callchain_trace_v5.sh

UPLOAD AFTER RUN
----------------
Upload BOTH:

  outputs/task39_corrected396_callchain_trace_v5.zip
  outputs/task39_corrected396_callchain_trace_v5.zip.sha256

Also paste the terminal summary.

DO NOT run Task 39 yet.
