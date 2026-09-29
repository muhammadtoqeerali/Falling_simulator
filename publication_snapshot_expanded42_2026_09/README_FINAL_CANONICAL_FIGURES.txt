FINAL CANONICAL PUBLICATION FIGURE GENERATOR

This is the final paper-facing simulator-figure stage.

Inputs
------
- corrected396 RNE-corrected campaign
- canonical publication-gate outputs
- existing paper evidence archive staging folder

It generates:
FIG_S01  aggregate validation check pass rates
FIG_S02  full validation-confidence distribution
FIG_S03  task-wise validation heatmap
FIG_S04  corrected396 simulated demographic-profile coverage
FIG_S05  representative sensor-model vs physics-truth acceleration
FIG_S06  representative sensor-model vs high-rate truth gyroscope
FIG_S07  representative event / impact timing
FIG_S08  recovered-impact summary by task (when canonical table supports it)
FIG_S09  canonical event-policy task summary (when canonical table supports it)

Representative profile selection is fixed before plotting:
ages 20, 30, 53, 63, 74, 78; task 21 preferred.
Selection is never based on confidence score, visual similarity, or favorable outcome.

All source files and SHA256 hashes are written to the output data/index files.

The script adds:
14_FINAL_CANONICAL_PUBLICATION_FIGURES/

to the evidence staging folder and rebuilds:
/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/paper_evidence_archive_FINAL.zip

No training, classifier inference, or simulator rerun is performed.

If you first want to inspect figures without rebuilding the ~GB master ZIP, run:
python build_final_canonical_publication_figures.py --no-zip
