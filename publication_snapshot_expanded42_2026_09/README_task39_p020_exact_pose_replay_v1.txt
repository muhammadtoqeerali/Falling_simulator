TASK 39 / P020 EXACT POSE REPLAY BUNDLE

Why this is now the correct next step
-------------------------------------
The project source already contains the instrumentation we were trying to add:

- MarkerKinematicsExporter:
  anatomical 3-D virtual markers, segment pose/velocity, joint qpos/qvel,
  TRC/MOT export, pose JSON and diagnostic visualizations.

- DynamicsContactAnalyzer:
  per-frame contact counts, GRF/CoP, support forces, primary impact body,
  tangential ratio and per-contact rows.

- PaperAlignmentExporter:
  OpenSim-style GRF, ExternalLoads XML, marker registration,
  multiview 3-D/2-D synthetic pose dataset and pose preview.

Therefore this wrapper does NOT patch scenario39_legacy.py.

What the wrapper adds
---------------------
It monkey-patches only MarkerKinematicsExporter.capture_frame at runtime to
record full MuJoCo state alongside the source's own exporter:

- qpos
- qvel
- all body xpos
- all body xquat
- subtree COM

It also attempts fixed-camera MuJoCo RGB renders at:
SETUP
FALL_ONSET
MAX_DESCENT
FIRST_CONTACT
PEAK_IMPACT
POST_IMPACT
REST

The exact phase times are read from the already-created:
outputs/task39_phase_and_replay_evidence_v1/03_PHASES/phase_timestamps.csv

Subject
-------
P020
age 49
height 1.63 m
female
weight 73.0 kg

CRITICAL SCIENTIFIC GATE
------------------------
The wrapper compares the replay against the final corrected396 P020 Task-39
trajectory.

It checks:
- pelvis-height RMSE <= 0.03 m
- 3-D sensor-position RMSE <= 0.04 m
- peak-impact timing difference <= 0.10 s
- total duration difference <= 0.15 s

If all pass:
USE_FOR_PAPER.json -> PASS

Then the replay-only marker/joint/contact/full-state products may be used as the
synchronized pose/contact visualization source while the corrected396 signal
remains the primary recorded trajectory.

If the gate fails:
USE_FOR_PAPER.json -> BLOCK

Do not use the replay pose renders in the paper. Upload the ZIP and we will
diagnose the exact discrepancy instead.

Run
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o task39_p020_exact_pose_replay_bundle_v1.zip

chmod +x run_task39_p020_exact_pose_replay_v1.sh

bash run_task39_p020_exact_pose_replay_v1.sh

The simulator environment is used:
  /mnt/hdd16T/ToqeerHomeBackup/mujoco_project/.venv/bin/python

NOT the Protechto CNN environment.

Expected important source-native outputs
----------------------------------------
*_markers.csv
*_segments.csv
*_joints.csv
*_marker_quality.csv
*_pose_markers.json
*_dynamics_frames.csv
*_contacts.csv
*_opensim_grf.mot
*_ExternalLoads.xml
*_marker_registration.json
*_synthetic_pose_dataset.json
*_synthetic_pose_preview.png

Added exact-state outputs
-------------------------
02_FULL_STATE/full_state_frames.npz
02_FULL_STATE/model_inventory.json

Phase-aligned outputs
---------------------
03_PHASE_ALIGNED/phase_markers_long.csv
03_PHASE_ALIGNED/phase_segments.csv
03_PHASE_ALIGNED/phase_joints.csv
03_PHASE_ALIGNED/phase_dynamics.csv
03_PHASE_ALIGNED/phase_contacts.csv
03_PHASE_ALIGNED/phase_state_indices.csv
03_PHASE_ALIGNED/phase_full_states.npz

Audits
------
04_AUDIT/source_identity.json
04_AUDIT/REPLAY_EQUIVALENCE.json
04_AUDIT/replay_export_inventory.json
04_AUDIT/render_capture_log.json

Upload after completion
-----------------------
outputs/task39_p020_exact_pose_replay_v1.zip
outputs/task39_p020_exact_pose_replay_v1.zip.sha256

Also paste:
Replay equivalence gate
Pelvis-height RMSE
Sensor-position 3D RMSE
Impact-time delta
Paper-use gate
