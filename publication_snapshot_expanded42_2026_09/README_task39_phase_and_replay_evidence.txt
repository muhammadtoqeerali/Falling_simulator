TASK 39 PHASE + REPLAY EVIDENCE COLLECTOR

The latest raw probe has established the real CSV format:
- ASCII text
- metadata preamble beginning with '#'
- ordinary comma-separated table after the preamble
- pandas reads it correctly with comment='#'

Observed main CSV columns include:
timestamp
accel_x/y/z
gyro_x/y/z
sensor_pos_x/y/z
sensor_vel_x/y/z
pelvis_height
impact_force
impact_magnitude
jerk_mag
fall_detected
accel_true_x/y/z
sensor_error_mag
soft_tissue_artifact
sensor_confidence

Observed high-rate-truth columns include:
timestamp
accel_true_x/y/z
gyro_true_x/y/z
accel_true_mag
gyro_true_mag
sensor_pos_x/y/z
sensor_vel_x/y/z
pelvis_height

What is NOT present in the observed CSV columns:
- qpos
- qvel
- joint positions
- body/marker xyz coordinates
- explicit anatomical skeleton

Therefore this collector:
1. parses all 22 final corrected Task-39 runs;
2. selects a transparent representative trial;
3. exports exact/derived scientific phases and all recorded curves;
4. scans the actual project source for the simulator runner, Task-39 logic,
   MuJoCo state/contact hooks, and corrected acceleration code;
5. packages relevant source excerpts/files so the next step can build an exact
   instrumented replay rather than guessing the simulator API.

RUN
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o task39_phase_and_replay_evidence_collector_v1.zip

chmod +x run_task39_phase_and_replay_evidence.sh

bash run_task39_phase_and_replay_evidence.sh

UPLOAD AFTER PASS
-----------------
outputs/task39_phase_and_replay_evidence_v1.zip
outputs/task39_phase_and_replay_evidence_v1.zip.sha256

Paste the terminal blocks:
- SELECTED REPRESENTATIVE PROFILE
- RECORDED POSE/STATE INVENTORY
- top 15 source files
- FIGURE READINESS
- final PASS
