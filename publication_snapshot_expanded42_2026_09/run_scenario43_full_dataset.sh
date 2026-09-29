#!/bin/bash

set -e


BASE="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"

cd "$BASE"


echo "============================================================"
echo " SCENARIO 43 FULL DATASET GENERATION"
echo "============================================================"


export DISPLAY=:1


echo
echo "[1/6] Running Scenario 43 simulation"
echo "============================================================"


python fall_dispatcher.py <<EOF2
43
55
1.75
male
78
EOF2



echo
echo "[2/6] Finding latest simulation output"
echo "============================================================"


LATEST=$(ls -td outputs/scenario43_* | head -1)

echo "LATEST:"
echo "$LATEST"



echo
echo "[3/6] Creating clean dataset folder"
echo "============================================================"


DATASET="outputs/scenario43_dataset"

mkdir -p "$DATASET"


RUN_NAME=$(basename "$LATEST")

FINAL="$DATASET/$RUN_NAME"


mkdir -p "$FINAL"



echo
echo "[4/6] Copying all simulation results"
echo "============================================================"


cp -r "$LATEST"/* "$FINAL"/



echo
echo "[5/6] Extracting video frames"
echo "============================================================"


mkdir -p "$FINAL/frames"


python tools/extract_video_frames.py \
"$FINAL/scenario43_native_render.mp4" \
"$FINAL/frames"



echo
echo "[6/6] Dataset verification"
echo "============================================================"


echo
echo "Dataset location:"
echo "$FINAL"


echo
echo "Files:"
find "$FINAL" -maxdepth 1 -type f | sed 's#^.*/##'



echo
echo "Frame count:"
find "$FINAL/frames" -name "*.png" | wc -l



echo
echo "Video information:"
ffprobe \
-v error \
-select_streams v:0 \
-show_entries stream=codec_name,width,height,nb_frames,duration \
-of default=noprint_wrappers=1 \
"$FINAL/scenario43_native_render.mp4"



echo
echo "============================================================"
echo " SCENARIO 43 DATASET COMPLETE"
echo "============================================================"

