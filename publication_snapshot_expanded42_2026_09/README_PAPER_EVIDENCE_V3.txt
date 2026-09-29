PAPER EVIDENCE V3

Purpose
-------
V2 collected a very broad evidence gallery. V3 adds a strict paper-facing layer so
old/superseded figures cannot be confused with canonical corrected validation.

It also fixes the age/diversity discovery issue by parsing the simulator's explicit
demographic fields from scenario/run paths, e.g.:

scenario42_age65_h1p75_sex_female_w78p0_...

Important distinction:
- these are SIMULATED demographic profiles;
- physical participant ages are reported separately only when explicit source metadata
  actually provides subject ID + age.

Main outputs inside:
13_CANONICAL_PAPER_EVIDENCE_V3/

- ALL_VALIDATION_ITEMS_WITH_STATUS.csv
- VALIDATION_STATUS_COUNTS.csv
- SIMULATED_PROFILE_FILES_WITH_DEMOGRAPHICS.csv
- SIMULATED_UNIQUE_PROFILE_DEMOGRAPHICS.csv
- SIMULATED_PROFILE_DIVERSITY_SUMMARY.csv
- AGE_STRATIFIED_SIMULATED_PROFILE_CANDIDATES.csv
- selected_profile_figures/
- PHYSICAL_SUBJECT_AGE_METADATA_DISCOVERED.csv
- PHYSICAL_DEMOGRAPHIC_SOURCE_SCAN.csv
- PAPER_FIGURE_CANDIDATES_CANONICALIZED.csv
- LEGACY_OR_SUPERSEDED_DO_NOT_USE_PRIMARY.csv
- fig_simulated_profile_age_distribution.{png,pdf,svg}
- fig_simulated_profile_age_sex_coverage.{png,pdf,svg}

The script performs no training or inference.

Final ZIP:
outputs/paper_evidence_archive_v1_FULL_V3.zip
