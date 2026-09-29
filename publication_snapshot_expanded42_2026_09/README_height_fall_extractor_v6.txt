HEIGHT-FALL PAPER EVIDENCE EXTRACTOR V6 — ROBUST

What the latest audit proved
----------------------------
- profile discovery is now real: 22 Task-39 profiles and 22 canonical manifest rows.
- P020 has the actual corrected trial directory.
- the previous selected file set was contaminated by a P014 run_manifest because
  metadata was allowed to override path identity.
- both corrected P020 CSV files exist, but ordinary pandas CSV parsing failed.
- the corrected P020 directory contains:
    fall_scenario39_age49_20260826_144048.csv
    fall_scenario39_age49_20260826_144048_highrate_truth.csv
    run_manifest.json
    validation.txt
- older v2 files also exist, but V6 stays on corrected396 only.

V6 fixes the actual issues:
---------------------------
1. Path identity is authoritative. P014 metadata can never become P020.
2. High-rate truth is recognized from filename.
3. CSV parsing probes encoding, delimiter, quoting, and whitespace layouts.
4. Every parser attempt is written to 00_AUDIT/parser_diagnostics.
5. A named time column is preferred.
6. If only frame/sample/step exists, time is reconstructed from run_manifest rate/dt.
7. Approx. 450 Hz is used only as an explicitly recorded final-project fallback for
   the corrected high-rate truth stream.
8. Vector-valued columns are inspected and 3-vectors are expanded to xyz.
9. qpos/qvel/body-state availability is audited so we know whether the final pose
   figure can be made directly from recorded data or needs an instrumented replay.

Run
---
cd /mnt/hdd16T/ToqeerHomeBackup/mujoco_project

unzip -o height_fall_paper_evidence_extractor_v6_robust.zip

chmod +x run_height_fall_extractor_v6.sh

bash run_height_fall_extractor_v6.sh

If PASS, upload
---------------
outputs/paper_task39_height_fall_evidence_v6.zip
outputs/paper_task39_height_fall_evidence_v6.zip.sha256

Most important terminal lines
-----------------------------
SELECTED task/profile
Robust CSV parsing
Primary timeseries
Time axis / Axis type
IMU columns
Kinematic columns
Contact/dynamics columns
State/qpos/qvel-like columns
Fall onset
Peak impact
Rest
Detected marker triplets
FIGURE READINESS
final PASS/ERROR

V6 modifies no existing experiment file and reruns no simulator/CNN process.
