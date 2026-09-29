HEIGHT-FALL EXTRACTOR V4 — ACTUAL-FUNCTION PATCH

Why V3 failed
-------------
V3 assumed the extractor contained:

    parse_task_profile_from_path()

but the actual extractor created in the first bundle contains:

    path_task_profile()

Therefore V3 correctly refused to modify the file because its expected function
boundaries did not exist.

V4 was built against the actual extractor structure.

What V4 changes
---------------
- patches `path_task_profile()`;
- patches `header_task_profile()` to normalize IDs such as 14 -> P014;
- adds `canonical_profile_id()`;
- explicitly prevents `mujoco_project` -> `roject`;
- syntax-checks before writing;
- compiles the actual patched extractor after writing;
- creates a backup before changes.

Run
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project
unzip -o height_fall_paper_evidence_extractor_v4_actual_patch.zip
chmod +x run_height_fall_extractor_v4.sh
bash run_height_fall_extractor_v4.sh

Success criteria
----------------
1. PATCH SELF-TEST: PASS
2. PATCH: PASS
3. TARGET COMPILE: PASS
4. TOP CANDIDATES contains real P### profiles with n_files > 0.
5. `roject` is absent.
6. AUTO selects a profile that has exact files.
7. A primary timeseries is found.
8. Final extraction reaches PASS.

If it still fails
-----------------
Do not patch again blindly.

Send:
outputs/paper_task39_height_fall_evidence_v4/00_AUDIT/task_file_index.csv
outputs/paper_task39_height_fall_evidence_v4/01_SELECTION/candidate_profiles.csv
and the terminal TOP CANDIDATES / ERROR lines.

That will let the next fix use the exact real filenames/schema rather than more assumptions.
