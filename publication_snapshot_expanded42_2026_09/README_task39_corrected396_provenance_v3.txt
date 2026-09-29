TASK39 CORRECTED396 PROVENANCE RECONCILIATION V3
================================================

PURPOSE
-------
This is the next step after Task-39 replay mismatch diagnostic V2 classified the
problem as SOURCE_OR_RUNTIME_VERSION_MISMATCH.

This bundle is deliberately STATIC / READ-ONLY with respect to the simulator,
scenario source, and canonical corrected396 data.

IT DOES NOT:
- import scenario39;
- import MuJoCo;
- import HumEnv / Meta Motivo / torch;
- run a simulation;
- modify scenario39 source;
- loosen the replay-equivalence thresholds;
- approve replay pose/render files for paper use.

IT DOES:
1. Fingerprint every plausible scenario39/task39 Python source variant.
2. Extract static timing/seed/version/export signatures without importing source.
3. Distinguish the canonical filename signature `fall_scenario39...` from the
   failed standalone replay signature `fall_forward_height...`.
4. Search likely campaign/runner/log files for the exact corrected396 output root.
5. Census all corrected396 run manifests for hidden source/runtime/version/seed/
   command/device/git/model fields.
6. Inspect zsh/bash/nohup/project logs for corrected396/scenario39 invocation clues.
7. Read git log/reflog metadata around August 2026.
8. Rank source candidates for HUMAN/AGENT REVIEW. Ranking is evidence triage,
   not proof and not an equivalence gate.

RUN
---
Place these three files in:
  /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

Then run:

  cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

  unzip -o task39_corrected396_provenance_v3_bundle.zip

  chmod +x run_task39_corrected396_provenance_v3.sh

  bash run_task39_corrected396_provenance_v3.sh

EXPECTED EARLY LINE
-------------------
  COMPILE: PASS

EXPECTED CORE OUTPUT
--------------------
  Root cause class: SOURCE_OR_RUNTIME_VERSION_MISMATCH
  Paper gate: BLOCK_REMAINS
  Simulator rerun performed: NO
  Scenario/Task39 source candidates: ...
  Campaign chain files with relevant hits: ...
  Files explicitly mentioning corrected396 root: ...
  Corrected396 manifests found: ... (expected 396)
  Task-39 manifests found: ... (expected 22)
  Shell/log history hits: ...
  Provenance proof level: ...
  Top static candidate: ...

UPLOAD AFTER RUN
----------------
Upload BOTH:
  outputs/task39_corrected396_provenance_v3.zip
  outputs/task39_corrected396_provenance_v3.zip.sha256

Also paste the terminal summary above.

IMPORTANT
---------
DO NOT run another Task-39 replay yet.

The next review must inspect:
  ROOT_PROVENANCE_DECISION.json
  06_RANKING/source_candidate_ranking.csv
  03_CAMPAIGN_CHAIN/exact_corrected396_contexts.txt
  05_HISTORY_GIT/campaign_history_hits.csv
  04_MANIFEST_CENSUS/task39_manifest_runtime_fields.csv
  02_SOURCE_CANDIDATES/scenario39_source_key_excerpts.txt

Only if that review proves the exact corrected396 entrypoint/source/runtime path
should a single new instrumented P020 replay be prepared.
