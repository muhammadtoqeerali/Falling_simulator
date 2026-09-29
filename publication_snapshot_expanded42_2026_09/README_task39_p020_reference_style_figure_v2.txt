TASK39 P020 REFERENCE-STYLE PUBLICATION FIGURE V2
=================================================

GOAL
----
Create a much clearer academic figure inspired by the visual grammar of the
user's attached Meta-Motivo/OpenSim/OpenPose paper example.

WHY THIS IS DIFFERENT FROM V1
-----------------------------
V1 was scientifically correct but too chart-heavy and visually difficult to
read. V2 gives the reader an immediate fall sequence first, then the exact
biomechanical/IMU evidence.

PANEL A
-------
Seven clean humanoid phase glyphs:
  Setup
  Fall onset
  Max descent
  First contact
  Peak impact
  Post-impact
  Rest

IMPORTANT:
The human glyphs are explicitly SCHEMATIC ONLY. They are not reconstructed
joints and must not be described as OpenPose/OpenSim kinematics.

PANELS B-F
----------
All quantitative content is exact corrected396 P020 evidence:
- 3-D lower-back sensor trajectory
- pelvis height
- impact force
- corrected acceleration magnitude
- corrected angular velocity magnitude
- exact seven-phase numerical table

NO failed replay skeleton/render/qpos is used.

OUTPUTS
-------
600-dpi PNG
vector PDF
vector SVG
exact phase-value CSV
caption
provenance JSON

RUN
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o task39_p020_reference_style_figure_v2_bundle.zip

chmod +x run_task39_p020_reference_style_figure_v2.sh

bash run_task39_p020_reference_style_figure_v2.sh

UPLOAD
------
outputs/task39_p020_reference_style_figure_v2.zip
outputs/task39_p020_reference_style_figure_v2.zip.sha256
