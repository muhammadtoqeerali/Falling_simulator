TASK39 REPLAY MISMATCH DIAGNOSTIC — READ ONLY

Why this is the correct next step
---------------------------------
The V2 headless replay worked technically:
- 432 source-native marker frames
- 432 full-state frames
- exact marker/joint/contact/state instrumentation available

But the replay-equivalence gate correctly BLOCKED paper use:
- pelvis-height RMSE ~0.93 m
- sensor-position RMSE ~1.01 m
- impact timing delta ~0.263 s

We must not loosen the gate or use those poses.

This diagnostic DOES NOT rerun MuJoCo. It uses the already-completed replay to
determine WHY it differs.

It tests:
1. canonical vs replay duration;
2. raw coordinate RMSE;
3. best constant coordinate offsets;
4. offset-corrected RMSE;
5. initial-position aligned RMSE;
6. best temporal shift +/-2 s;
7. canonical/replay run-manifest differences;
8. every scenario39*.py source variant and backup;
9. every project file referring to corrected396 / RNE / high-rate campaign;
10. exact campaign code excerpts needed to reproduce the original runtime.

Run
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o task39_replay_mismatch_diagnostic_bundle_v1.zip

chmod +x run_task39_replay_mismatch_diagnostic_v1.sh

bash run_task39_replay_mismatch_diagnostic_v1.sh

Upload after PASS
-----------------
outputs/task39_replay_mismatch_diagnostic_v1.zip
outputs/task39_replay_mismatch_diagnostic_v1.zip.sha256

Paste:
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

Do NOT run another replay before this diagnostic is reviewed.
