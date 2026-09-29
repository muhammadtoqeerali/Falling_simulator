TASK 39 / P020 EXACT POSE REPLAY V2 — HEADLESS

Reason for V2
-------------
V1 reached the real Task-39 simulation loop but stopped at:

    GLFWError: X11: The DISPLAY environment variable is missing
    ERROR: could not initialize GLFW

The active scenario39_legacy.py uses:

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

and later calls viewer.sync(). The passive viewer is a visualization/UI
dependency, not the physics integrator.

V2 therefore replaces ONLY `mujoco.viewer.launch_passive` in memory with a
no-window viewer that provides the camera fields and sync()/context-manager API
the scenario expects.

It does NOT modify scenario39_legacy.py on disk.

Unchanged scientific/runtime path
---------------------------------
- Meta Motivo embeddings
- HumEnv environment/model/data
- subject anthropometry
- Task39HeightLayer
- EnhancedBiofidelicController
- env stepping / MuJoCo dynamics
- high-rate bridge if active in the source
- IMUValidator
- MarkerKinematicsExporter
- DynamicsContactAnalyzer
- PaperAlignmentExporter
- qpos/qvel/xpos/xquat capture
- replay-equivalence gate

Optional phase RGB rendering
----------------------------
The launcher requests:

    MUJOCO_GL=egl

for offscreen MuJoCo Renderer images. If EGL rendering is unavailable, render
errors are logged and the exact state/marker/contact evidence still remains
usable. Publication use is controlled by REPLAY_EQUIVALENCE.json, not by PNG
availability.

Run
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o task39_p020_exact_pose_replay_bundle_v2_headless.zip

chmod +x run_task39_p020_exact_pose_replay_v2_headless.sh

bash run_task39_p020_exact_pose_replay_v2_headless.sh

Expected early confirmation
---------------------------
COMPILE: PASS
[HeadlessReplay] passive MuJoCo viewer replaced with no-op viewer
[HeadlessReplay] physics/controller/exporter path unchanged

Expected final evidence
-----------------------
Source-native marker frames:
Captured full-state frames:
Replay equivalence gate:
Pelvis-height RMSE [m]:
Sensor-position 3D RMSE [m]:
Impact-time delta [s]:
Paper-use gate:

Upload after completion
-----------------------
outputs/task39_p020_exact_pose_replay_v2_headless.zip
outputs/task39_p020_exact_pose_replay_v2_headless.zip.sha256

If it fails, paste from "[4/4] Running simulation..." through the traceback/error.
