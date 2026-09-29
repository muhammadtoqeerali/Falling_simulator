HEIGHT-FALL PAPER EVIDENCE EXTRACTOR

Default: Task 39 = Forward fall from height.
Use Task 40 for backward fall from height.

The extractor does NOT rerun the simulator. It selects one exact corrected396 Task-39 trial,
copies its source files, derives figure stages from recorded onset/impact/physics data, and exports:
- candidate profile ranking;
- exact source files + SHA256;
- exact phase timestamps;
- phase snapshot rows;
- marker/skeleton XYZ coordinates when available;
- kinematic timeseries;
- contact/impact timeseries;
- lower-back IMU timeseries;
- diagnostic phase previews;
- one ZIP for upload to ChatGPT.

Run from /mnt/hdd16T/ToqeerHomeBackup/mujoco_project after unzipping the bundle:
  chmod +x run_height_fall_paper_evidence.sh
  bash run_height_fall_paper_evidence.sh

Upload afterward:
  outputs/paper_task39_height_fall_evidence_v1.zip
  outputs/paper_task39_height_fall_evidence_v1.zip.sha256

If you want a specific candidate profile after inspecting candidate_profiles.csv, rerun:
  /mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python -u prepare_height_fall_paper_evidence.py --task-id 39 --profile-id <PROFILE_ID>

Diagnostic PNGs are NOT the final publication artwork.
