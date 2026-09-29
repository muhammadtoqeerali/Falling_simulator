TASK39 REPLAY MISMATCH DIAGNOSTIC V2 — READ ONLY

V1 failure explained
--------------------
V1 did not fail because replay data lacked `timestamp`.

The replay package contains the correct main file:

    fall_forward_height_age49_20260901_085038.csv

with the normal 27-column stream including:
    timestamp
    sensor_pos_x/y/z
    pelvis_height
    impact_force
    impact_magnitude
    ...

But V1 used a broad filename glob and accidentally selected:

    *_marker_quality.csv

because that file was not excluded and sorted after the actual main CSV.
That produced KeyError('timestamp').

V2 fixes file resolution by:
1. reading 04_AUDIT/replay_export_inventory.json first;
2. validating the candidate's schema;
3. falling back only to an exact timestamped-main filename regex;
4. never choosing a replay file by lexical sort alone.

No MuJoCo rerun
---------------
This remains a read-only diagnostic. It examines:
- final corrected396 P020 Task-39 canonical CSV;
- already-completed V2 headless replay;
- source versions/backups;
- corrected396 campaign/RNE/high-rate provenance.

Run
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o task39_replay_mismatch_diagnostic_bundle_v2.zip

chmod +x run_task39_replay_mismatch_diagnostic_v2.sh

bash run_task39_replay_mismatch_diagnostic_v2.sh

Expected early output
---------------------
COMPILE: PASS
Canonical main CSV: ...
Resolved replay main CSV: .../fall_forward_height_age49_<timestamp>.csv
Canonical columns: 27 | Replay columns: 27

Then expect:
Canonical duration
Replay duration
Duration delta
Raw pelvis RMSE
Offset-corrected pelvis RMSE
Raw sensor-z RMSE
Offset-corrected sensor-z RMSE
Best time shift
ROOT CAUSE CLASS
Scenario39 source variants found
Campaign provenance files found

Upload after PASS
-----------------
outputs/task39_replay_mismatch_diagnostic_v2.zip
outputs/task39_replay_mismatch_diagnostic_v2.zip.sha256

Do not perform another replay before this V2 diagnostic is reviewed.
