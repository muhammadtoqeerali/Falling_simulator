#!/usr/bin/env bash
set -u

PROJECT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
PY="/mnt/hdd16T/ToqeerHomeBackup/toqeer/protechto_env/bin/python"

cd "$PROJECT" || exit 1

echo "======================================================================"
echo "1) PATCH BASE BUILDER TO REMOVE OPTIONAL tabulate DEPENDENCY"
echo "======================================================================"

"$PY" -u patch_paper_evidence_builder_no_tabulate.py || exit 2

echo
echo "======================================================================"
echo "2) COMPILE BUILDERS"
echo "======================================================================"

"$PY" -m py_compile \
  build_paper_evidence_archive.py \
  augment_paper_evidence_archive.py || exit 3

echo
echo "======================================================================"
echo "3) BUILD BASE FULL EVIDENCE ARCHIVE INCLUDING CHECKPOINTS"
echo "======================================================================"

"$PY" -u build_paper_evidence_archive.py \
  --output-dir "$PROJECT/outputs/paper_evidence_archive_v1" \
  --zip-path "$PROJECT/outputs/paper_evidence_archive_v1_BASE.zip" \
  --include-checkpoints || exit 4

echo
echo "======================================================================"
echo "4) ADD SIMULATOR VALIDATION GALLERY + AGE/SUBJECT DIVERSITY"
echo "======================================================================"

"$PY" -u augment_paper_evidence_archive.py \
  --stage "$PROJECT/outputs/paper_evidence_archive_v1" \
  --zip "$PROJECT/outputs/paper_evidence_archive_v1_FULL.zip" || exit 5

echo
echo "======================================================================"
echo "FULL PAPER EVIDENCE ARCHIVE COMPLETE"
echo "======================================================================"
echo "Final ZIP:"
echo "$PROJECT/outputs/paper_evidence_archive_v1_FULL.zip"
echo
echo "Inspect key inventories:"
echo "$PROJECT/outputs/paper_evidence_archive_v1/09_SIMULATOR_VALIDATION_GALLERY/VALIDATION_FIGURE_AND_DATA_CATALOG.csv"
echo "$PROJECT/outputs/paper_evidence_archive_v1/10_SUBJECT_AND_PROFILE_DIVERSITY/AGE_STRATIFIED_SUBJECT_PROFILE_CANDIDATES.csv"
echo "$PROJECT/outputs/paper_evidence_archive_v1/11_PAPER_FIGURE_SHORTLIST/FIGURE_SHORTLIST.csv"
