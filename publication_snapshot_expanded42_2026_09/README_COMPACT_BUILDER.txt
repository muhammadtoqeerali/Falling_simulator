COMPACT PAPER-WRITING EVIDENCE ARCHIVE BUILDER

Keeps manuscript-relevant results, figures, plots, classification/confusion reports,
canonical simulator validation summaries, event metadata, profile/task details, units,
labels, filtering/windowing/splitting facts, and provenance notes.

Excludes code, checkpoints, raw datasets, raw corrected trial CSVs, high-rate truth CSVs,
NPY tensors, large y_true/y_pred arrays, and broad superseded galleries.

IMPORTANT: the accepted final Protechto npseg pipeline does NOT actively remove the
last 150 ms. The old -15 sample lines are commented out. The accepted synthetic pipeline
also has no manual terminal 150-ms trim. The compact archive states this explicitly.

Final outputs:
  outputs/paper_writing_evidence_COMPACT.zip
  outputs/paper_writing_evidence_COMPACT.zip.sha256

Use this compact ZIP for manuscript writing. Keep the ~2.6GB final archive only as an
optional reproducibility backup.
