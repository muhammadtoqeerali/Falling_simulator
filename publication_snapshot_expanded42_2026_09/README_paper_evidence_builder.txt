Paper Evidence Archive Builder

Files:
- build_paper_evidence_archive.py
- build_paper_evidence_archive.sh

Copy both files into:
/mnt/hdd16T/ToqeerHomeBackup/mujoco_project

Then run:
bash build_paper_evidence_archive.sh

The builder does NOT train models or run inference.
It verifies the 35 fold-completion markers, collects exact method/source snapshots,
final results, classification reports, confusion matrices, simulator validation evidence,
paper-ready tables, plot data, and publication figures, writes SHA256 manifests, and
creates:
/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/paper_evidence_archive_v1.zip

By default it DOES NOT copy the 35 checkpoint binaries into the ZIP; it records their
original path, size, and SHA256 in CHECKPOINT_MANIFEST.csv. To create a much larger
full-weight archive, run the Python script manually with --include-checkpoints.

Primary paper reporting policy encoded in the builder:
- Segment: REAL_ONLY, SIM_ONLY, MIX20, MIX50, MIX70, MIX100.
- Event physical: REAL_ONLY, MIX20, MIX50, MIX70, MIX100.
- SIM_ONLY event: sensitivity only (177/310 = 57.10% in the current run).
- EXP02 substitution: archived under excluded diagnostics, not in primary tables.
