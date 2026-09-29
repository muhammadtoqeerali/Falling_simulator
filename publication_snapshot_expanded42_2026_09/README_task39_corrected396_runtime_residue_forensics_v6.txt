TASK39 CORRECTED396 RUNTIME-RESIDUE FORENSICS V6
================================================

WHY V6
------
V5 correctly remained unresolved.

The useful V5 facts are:
- selected pre-campaign scenario39_legacy.py is different from the failed replay source;
- two filesystem-era controller references existed;
- Git retained ZERO historical controller versions for those references;
- historical scenario39_legacy.py contains fall_forward_height;
- it does NOT contain mj_rnePostConstraint.

That combination strongly suggests the missing provenance may live in an
untracked/ephemeral campaign wrapper, high-rate sidecar patch, compiled Python
artifact, log, backup, or editor history rather than in scenario source Git
history.

V6 therefore searches historical runtime residue.

WHAT V6 INSPECTS
----------------
1. Project launcher/config/log/backup files, with special emphasis on files
   modified around 24–28 August 2026.
2. Python .pyc files. It uses marshal only to READ code-object metadata,
   filenames, names, and string constants. IT NEVER EXECUTES THE BYTECODE.
3. Git reflog, stash, and unreachable commits.
4. Common VS Code/VSCodium/code-server/local-history locations.
5. Parent-level corrected396/high-rate campaign provenance.
6. All 22 canonical Task-39 main CSV durations as a timing fingerprint.

SAFETY
------
- NO scenario39 import.
- NO MuJoCo import.
- NO HumEnv / Meta Motivo / torch import.
- NO simulator execution.
- NO bytecode execution.
- NO source modification.
- NO replay authorization.
- Paper gate remains BLOCKED.

RUN EXACTLY
-----------
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o task39_corrected396_runtime_residue_forensics_v6_bundle.zip

chmod +x run_task39_corrected396_runtime_residue_v6.sh

bash run_task39_corrected396_runtime_residue_v6.sh

UPLOAD AFTER RUN
----------------
Upload BOTH:

  outputs/task39_corrected396_runtime_residue_forensics_v6.zip
  outputs/task39_corrected396_runtime_residue_forensics_v6.zip.sha256

Also paste the terminal summary.

DO NOT run Task 39 yet.
