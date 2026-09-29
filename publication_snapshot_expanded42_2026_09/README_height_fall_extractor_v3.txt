HEIGHT-FALL EXTRACTOR V3 SAFE PATCH

Why V2 failed
-------------
The V2 patch file itself was malformed. It contained:

    NEW_FUNCTION = r

instead of a valid multiline string, so Python raised:

    NameError: name 'r' is not defined

V3 fixes the patch script itself and adds safety checks before modifying the extractor.

What V3 does
------------
1. Self-tests the replacement parser on P014/P015/P016 examples.
2. Explicitly verifies that `mujoco_project` cannot become profile `roject`.
3. Backs up the current extractor.
4. Replaces only `parse_task_profile_from_path()`.
5. Compiles the patched extractor immediately.
6. Reruns Task 39 with AUTO profile selection.
7. Writes a fresh V3 output folder and ZIP.

Run
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project
unzip -o height_fall_paper_evidence_extractor_v3_safe_patch.zip
chmod +x run_height_fall_extractor_v3.sh
bash run_height_fall_extractor_v3.sh

Do not type angle-bracket placeholders literally.
For a specific profile later, use for example:

    --profile-id P014

not:

    --profile-id <PROFILE_ID>

What to paste back
------------------
Please paste only:
- TOP CANDIDATES
- SELECTED task/profile
- Primary timeseries
- Fall onset
- Peak impact
- Rest
- Detected marker triplets
- final PASS/ERROR lines

If PASS, upload:
outputs/paper_task39_height_fall_evidence_v3.zip
outputs/paper_task39_height_fall_evidence_v3.zip.sha256
