#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"

echo "============================================================================================"
echo "TASK39 P020 CANONICAL PUBLICATION FIGURE V1"
echo "============================================================================================"

"$PY" -m py_compile build_task39_p020_canonical_publication_figure_v1.py
echo "COMPILE: PASS"

"$PY" build_task39_p020_canonical_publication_figure_v1.py

OUT="outputs/task39_p020_canonical_publication_figure_v1"
ZIP="outputs/task39_p020_canonical_publication_figure_v1.zip"

rm -f "$ZIP" "$ZIP.sha256"
zip -qr "$ZIP" "$OUT"
sha256sum "$ZIP" > "$ZIP.sha256"

echo
echo "FINAL FILES:"
ls -lh "$ZIP" "$ZIP.sha256"
cat "$ZIP.sha256"
