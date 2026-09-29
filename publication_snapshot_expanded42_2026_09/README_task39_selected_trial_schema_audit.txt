TASK 39 SELECTED-TRIAL SCHEMA AUDIT

Why this step exists
--------------------
V5 successfully fixed profile discovery:
- 88 Task-39 files
- 22 event-manifest rows
- P020 selected with 4 exact files
- no fake `roject` profile

V5 then stopped because none of the selected CSVs had a time column matching the
extractor's current aliases.

Do NOT guess another time-column name yet.

This audit inspects the actual P020 files and reports:
- exact paths
- row/column counts
- all column names
- numeric monotonic candidates (time/frame/sample/index)
- IMU columns
- pelvis/trunk/kinematic columns
- contact/force/impact columns
- qpos/qvel/state columns
- event columns
- JSON keys
- nearby Task-39/P020 files under outputs/_highrate_overnight

Run
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project
unzip -o task39_selected_trial_schema_audit_bundle_v1.zip
chmod +x run_task39_selected_trial_schema_audit.sh
bash run_task39_selected_trial_schema_audit.sh

Then upload:
outputs/task39_selected_trial_schema_audit_v1.zip
outputs/task39_selected_trial_schema_audit_v1.zip.sha256

This audit modifies nothing and reruns nothing.
