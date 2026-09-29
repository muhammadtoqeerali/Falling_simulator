HEIGHT-FALL EXTRACTOR V2 PATCH

The first extractor run produced a false profile:

    profile_id = roject

because the generic profile regex matched the letter "p" in:

    mujoco_project

As a result, all 66 Task-39 files were incorrectly grouped under `roject`,
while the canonical event manifest correctly contained P014, P015, etc.
AUTO then selected P014 from the manifest, but P014 had zero associated files.

This patch fixes only the profile/task path parser.

Run:

    cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project
    unzip -o height_fall_paper_evidence_extractor_v2_patch.zip
    chmod +x run_height_fall_extractor_v2.sh
    bash run_height_fall_extractor_v2.sh

Expected improvement:
- `roject` must disappear.
- Task-39 files should resolve to canonical P### profiles.
- The top candidate should have actual files (normally ~3 per profile if 66 files = 22 profiles x 3 artifacts).
- AUTO can then select a profile that has both canonical event-manifest eligibility and exact simulator files.

IMPORTANT:
Do NOT type:

    --profile-id <PROFILE_ID>

literally in zsh.

Angle brackets are shell redirection syntax.

If you later want a specific profile, use an actual ID, for example:

    --profile-id P014

only after `candidate_profiles.csv` confirms that P014 has exact files.

Upload after PASS:
    outputs/paper_task39_height_fall_evidence_v2.zip
    outputs/paper_task39_height_fall_evidence_v2.zip.sha256
