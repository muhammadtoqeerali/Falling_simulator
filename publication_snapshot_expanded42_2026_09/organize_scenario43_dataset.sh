#!/bin/bash

set -e


BASE="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"

cd "$BASE"


echo "============================================================"
echo " SCENARIO 43 DATASET ORGANIZATION"
echo "============================================================"


LATEST=$(ls -td outputs/scenario43_dataset/scenario43_* | head -1)

echo "SOURCE:"
echo "$LATEST"


FINAL="$LATEST/organized_dataset"


mkdir -p "$FINAL"/{
video,
frames,
imu,
kinematics,
dynamics,
opensim,
metadata,
visualizations
}



echo
echo "[1/7] Organizing video"
echo "============================================================"


cp "$LATEST"/scenario43_native_render.mp4 \
"$FINAL/video/"



echo
echo "[2/7] Organizing frames"
echo "============================================================"


cp -r "$LATEST"/frames \
"$FINAL/"



echo
echo "[3/7] Organizing IMU data"
echo "============================================================"


cp "$LATEST"/*highrate*.csv \
"$FINAL/imu/" 2>/dev/null || true

cp "$LATEST"/*.csv \
"$FINAL/imu/" 2>/dev/null || true



echo
echo "[4/7] Organizing kinematics"
echo "============================================================"


cp "$LATEST"/*markers* \
"$FINAL/kinematics/" 2>/dev/null || true


cp "$LATEST"/*segments* \
"$FINAL/kinematics/" 2>/dev/null || true


cp "$LATEST"/*joints* \
"$FINAL/kinematics/" 2>/dev/null || true



echo
echo "[5/7] Organizing dynamics"
echo "============================================================"


cp "$LATEST"/*contacts* \
"$FINAL/dynamics/" 2>/dev/null || true


cp "$LATEST"/*dynamics* \
"$FINAL/dynamics/" 2>/dev/null || true


cp "$LATEST"/*cop* \
"$FINAL/dynamics/" 2>/dev/null || true



echo
echo "[6/7] Organizing OpenSim files"
echo "============================================================"


cp "$LATEST"/*.mot \
"$FINAL/opensim/" 2>/dev/null || true


cp "$LATEST"/*.xml \
"$FINAL/opensim/" 2>/dev/null || true



echo
echo "[7/7] Organizing metadata and visualizations"
echo "============================================================"


cp "$LATEST"/run_manifest.json \
"$FINAL/metadata/"


cp "$LATEST"/*validation* \
"$FINAL/metadata/" 2>/dev/null || true


cp "$LATEST"/*.png \
"$FINAL/visualizations/" 2>/dev/null || true



echo
echo "============================================================"
echo "GENERATING DATASET INDEX"
echo "============================================================"


find "$FINAL" -type f | sort \
> "$FINAL/metadata/file_index.txt"



echo
echo "============================================================"
echo "GENERATING FRAME TIMESTAMP FILE"
echo "============================================================"


python - <<PY

from pathlib import Path
import csv

folder = Path("$FINAL/frames")

files = sorted(folder.glob("*.png"))

with open(
    "$FINAL/metadata/frame_timestamps.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "frame_id",
            "filename",
            "time_seconds",
            "fps"
        ]
    )


    for i,x in enumerate(files):

        writer.writerow(
            [
                i,
                x.name,
                i/30.0,
                30
            ]
        )

print("Frames indexed:",len(files))

PY



echo
echo "============================================================"
echo "GENERATING CHECKSUM"
echo "============================================================"


cd "$FINAL"

find . -type f -exec sha256sum {} \; \
> metadata/SHA256SUMS.txt



echo
echo "============================================================"
echo "DATASET COMPLETE"
echo "============================================================"


echo
echo "FINAL DATASET:"
echo "$FINAL"


echo
echo "SUMMARY"

echo "Frames:"
find frames -name "*.png" | wc -l


echo "Files:"
find . -type f | wc -l


