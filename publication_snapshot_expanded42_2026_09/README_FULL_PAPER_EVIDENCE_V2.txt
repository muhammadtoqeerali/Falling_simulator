FULL PAPER EVIDENCE ARCHIVE V2

This package fixes the previous pandas.to_markdown/tabulate failure without installing
anything into the Protechto environment.

It also extends the archive with:

09_SIMULATOR_VALIDATION_GALLERY
- simulator validation figures, plots, charts and relevant tables discovered from the
  publication-gate audit, corrected396 campaign, and project outputs
- source path, validation category, size and SHA256 for every copied item

10_SUBJECT_AND_PROFILE_DIVERSITY
- automatically discovered subject/profile demographics
- age-band coverage
- representative age-stratified subjects/profiles
- matches to existing validation files when subject/profile IDs occur in filenames/paths
- no ages are inferred; only explicit age metadata are used

11_PAPER_FIGURE_SHORTLIST
- convenience shortlist based on validation category and publication-friendly format
- not based on favorable result values

12_CLAIM_TO_EVIDENCE_MAP
- paper claim/section -> exact evidence location

The full wrapper builds the base archive WITH all checkpoint binaries, adds the extended
simulator validation evidence, then creates:

/mnt/hdd16T/ToqeerHomeBackup/mujoco_project/outputs/paper_evidence_archive_v1_FULL.zip

No model training or inference is performed.
