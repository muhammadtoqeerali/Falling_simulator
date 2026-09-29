#!/bin/bash

set -e


BASE="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"

cd "$BASE"


echo "============================================================"
echo " AUTOMATIC FALL DATASET ORGANIZER"
echo " ALL SCENARIOS"
echo "============================================================"



for SRC in outputs/scenario*_dataset/scenario*; do


    if [ ! -d "$SRC" ]; then
        continue
    fi


    echo
    echo "============================================================"
    echo "PROCESSING:"
    echo "$SRC"
    echo "============================================================"



    FINAL="$SRC/organized_dataset"



    mkdir -p "$FINAL/video"
    mkdir -p "$FINAL/frames"
    mkdir -p "$FINAL/imu"
    mkdir -p "$FINAL/kinematics"
    mkdir -p "$FINAL/dynamics"
    mkdir -p "$FINAL/opensim"
    mkdir -p "$FINAL/metadata"
    mkdir -p "$FINAL/visualizations"



    echo "[1] Video"

    cp "$SRC"/*.mp4 \
    "$FINAL/video/" \
    2>/dev/null || true



    echo "[2] Frames"


    if [ -d "$SRC/frames" ]; then

        cp -r "$SRC/frames"/* \
        "$FINAL/frames/" \
        2>/dev/null || true

    fi



    echo "[3] IMU"


    cp "$SRC"/*imu*.csv \
    "$FINAL/imu/" \
    2>/dev/null || true


    cp "$SRC"/*highrate*.csv \
    "$FINAL/imu/" \
    2>/dev/null || true



    echo "[4] Kinematics"


    cp "$SRC"/*markers* \
    "$FINAL/kinematics/" \
    2>/dev/null || true


    cp "$SRC"/*segments* \
    "$FINAL/kinematics/" \
    2>/dev/null || true


    cp "$SRC"/*joints* \
    "$FINAL/kinematics/" \
    2>/dev/null || true



    echo "[5] Dynamics"


    cp "$SRC"/*contacts* \
    "$FINAL/dynamics/" \
    2>/dev/null || true


    cp "$SRC"/*dynamics* \
    "$FINAL/dynamics/" \
    2>/dev/null || true


    cp "$SRC"/*cop* \
    "$FINAL/dynamics/" \
    2>/dev/null || true



    echo "[6] OpenSim"


    cp "$SRC"/*.mot \
    "$FINAL/opensim/" \
    2>/dev/null || true


    cp "$SRC"/*.xml \
    "$FINAL/opensim/" \
    2>/dev/null || true



    echo "[7] Metadata"


    cp "$SRC"/run_manifest.json \
    "$FINAL/metadata/" \
    2>/dev/null || true


    cp "$SRC"/*validation* \
    "$FINAL/metadata/" \
    2>/dev/null || true



    echo "[8] Visualizations"


    cp "$SRC"/*.png \
    "$FINAL/visualizations/" \
    2>/dev/null || true




    echo "[9] File index"


    find "$FINAL" \
    -type f \
    | sort \
    > "$FINAL/metadata/file_index.txt"



    echo "[10] Frame timestamps"


    python - <<PY

from pathlib import Path
import csv

folder = Path("$FINAL/frames")

files = sorted(folder.glob("*.png"))

if files:

    with open(
        "$FINAL/metadata/frame_timestamps.csv",
        "w",
        newline=""
    ) as f:

        w = csv.writer(f)

        w.writerow(
            [
            "frame_id",
            "filename",
            "time_seconds",
            "fps"
            ]
        )


        for i,x in enumerate(files):

            w.writerow(
                [
                i,
                x.name,
                i/30.0,
                30
                ]
            )

    print("Frames indexed:",len(files))

PY




    echo "[11] SHA256"


    (
    cd "$FINAL"
    find . -type f -exec sha256sum {} \;
    ) > "$FINAL/metadata/SHA256SUMS.txt"




    echo
    echo "DONE:"
    echo "$FINAL"


done



echo
echo "============================================================"
echo " ALL DATASETS ORGANIZED"
echo "============================================================"

