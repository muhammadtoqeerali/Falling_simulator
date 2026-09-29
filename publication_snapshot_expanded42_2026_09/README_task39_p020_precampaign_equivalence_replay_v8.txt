TASK39 P020 PRE-CAMPAIGN SOURCE EQUIVALENCE REPLAY V8
=====================================================

V8 is a MINIMAL TECHNICAL FIX to V7.

V7 did not perform a scientific replay. Its driver resolver accidentally chose:
  run_task39_p020_precampaign_equivalence_replay_v7.py

That caused V7 to recursively execute itself and alter its own GIT_SOURCE
literal in memory. The historical source hash had already passed before this
technical failure.

V8 changes only replay-driver resolution:
- explicitly excludes V7/V8 historical-equivalence launchers;
- first looks for the original exact-pose V2 headless launcher by exact name;
- otherwise requires multiple V2 instrumentation signatures;
- strongly prefers exact_pose_replay + v2 + headless;
- has a hard guard preventing self-selection.

Nothing scientific changes:
- same historical Git source;
- same source SHA256;
- same P020 canonical corrected396 reference;
- same V2 full-state/marker/contact/headless instrumentation;
- same strict thresholds;
- canonical corrected396 signals remain primary.

RUN EXACTLY
-----------
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o task39_p020_precampaign_equivalence_replay_v8_resolver_fix_bundle.zip

chmod +x run_task39_p020_precampaign_equivalence_replay_v8.sh

bash run_task39_p020_precampaign_equivalence_replay_v8.sh

UPLOAD AFTER RUN
----------------
Upload:
  outputs/task39_p020_precampaign_equivalence_replay_v8.zip
  outputs/task39_p020_precampaign_equivalence_replay_v8.zip.sha256

Paste the V8 FINAL DECISION block.

Exit:
  0  = strict PASS
 10  = scientific FAIL
  5  = technical error
