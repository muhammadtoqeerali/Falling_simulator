TASK39 P020 CANONICAL PUBLICATION FIGURE V1
===========================================

This is the fast, scientifically defensible figure path after the V8 historical
replay failed the strict equivalence gate.

It intentionally DOES NOT use:
- failed replay MuJoCo renders;
- replay qpos/qvel;
- replay anatomical skeleton;
- replay contact identity/GRF.

It uses ONLY exact final corrected396 P020 Task-39 files and the locked seven
canonical phase times.

The generated figure contains:
(a) canonical lower-back sensor world trajectory with all seven phase markers;
(b) pelvis height + sensor speed;
(c) impact force + jerk;
(d) corrected physics-truth acceleration;
(e) corrected physics-truth angular velocity;
(f) exact seven-phase state matrix.

Outputs are produced in:
  PNG at 600 dpi
  vector PDF
  vector SVG
  exact phase-summary CSV
  figure caption TXT
  provenance JSON

RUN
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o task39_p020_canonical_publication_figure_v1_bundle.zip

chmod +x run_task39_p020_canonical_publication_figure_v1.sh

bash run_task39_p020_canonical_publication_figure_v1.sh

UPLOAD AFTER RUN
----------------
Upload:
  outputs/task39_p020_canonical_publication_figure_v1.zip
  outputs/task39_p020_canonical_publication_figure_v1.zip.sha256

Then the figure can be visually reviewed and, if needed, polished without
changing any scientific data.
