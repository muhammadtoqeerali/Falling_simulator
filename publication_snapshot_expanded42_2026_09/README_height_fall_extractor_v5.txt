HEIGHT-FALL PAPER EVIDENCE EXTRACTOR V5 — STANDALONE

This version does NOT patch the old extractor.

Why
---
V2/V4 patch bundles were malformed before execution, and V3 targeted the wrong
function name. V5 avoids the entire patching path and is a fresh self-contained
extractor.

Run
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o height_fall_paper_evidence_extractor_v5_standalone.zip

chmod +x run_height_fall_extractor_v5.sh

bash run_height_fall_extractor_v5.sh

Success criteria
----------------
- PARSER SELF-TEST: PASS
- COMPILE: PASS
- no fake `roject` profile
- TOP CANDIDATES contains file-backed P### profiles
- SELECTED task/profile has nonzero files
- Primary timeseries found
- phase timestamps exported
- final PASS

If PASS, upload
---------------
outputs/paper_task39_height_fall_evidence_v5.zip
outputs/paper_task39_height_fall_evidence_v5.zip.sha256

If it fails after producing the audit folder, provide:
outputs/paper_task39_height_fall_evidence_v5/00_AUDIT/task_file_index.csv
outputs/paper_task39_height_fall_evidence_v5/01_SELECTION/candidate_profiles.csv
plus the terminal error lines.

Do not type <PROFILE_ID> literally in zsh.
Use a real value such as P014 only if you later choose a specific profile.
