#!/bin/bash

set -e


BASE="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"

cd "$BASE"


REPORT="outputs/all_dataset_validation_report.txt"


echo "============================================================" > "$REPORT"
echo " FALL SIMULATION DATASET VALIDATION REPORT" >> "$REPORT"
echo " Generated: $(date)" >> "$REPORT"
echo "============================================================" >> "$REPORT"


echo "" >> "$REPORT"


TOTAL=0


for DATASET in outputs/scenario*_dataset/scenario*/organized_dataset
do

    if [ ! -d "$DATASET" ]; then
        continue
    fi


    TOTAL=$((TOTAL+1))


    echo "============================================================" >> "$REPORT"
    echo "DATASET:" >> "$REPORT"
    echo "$DATASET" >> "$REPORT"
    echo "============================================================" >> "$REPORT"



    echo "" >> "$REPORT"
    echo "FILES:" >> "$REPORT"

    find "$DATASET" -type f | wc -l >> "$REPORT"



    echo "" >> "$REPORT"
    echo "FRAMES:" >> "$REPORT"

    if [ -d "$DATASET/frames" ]; then
        find "$DATASET/frames" -name "*.png" | wc -l >> "$REPORT"
    else
        echo "MISSING" >> "$REPORT"
    fi



    echo "" >> "$REPORT"
    echo "VIDEO:" >> "$REPORT"


    VIDEO=$(find "$DATASET/video" -name "*.mp4" | head -1)


    if [ -f "$VIDEO" ]; then

        ffprobe \
        -v error \
        -select_streams v:0 \
        -show_entries stream=codec_name,width,height,nb_frames,duration \
        -of default=noprint_wrappers=1 \
        "$VIDEO" >> "$REPORT"

    else

        echo "NO VIDEO" >> "$REPORT"

    fi



    echo "" >> "$REPORT"
    echo "CORE FILE CHECK:" >> "$REPORT"



    for F in \
    metadata/run_manifest.json \
    metadata/frame_timestamps.csv \
    metadata/SHA256SUMS.txt

    do

        if [ -f "$DATASET/$F" ]; then
            echo "OK  $F" >> "$REPORT"
        else
            echo "MISS $F" >> "$REPORT"
        fi

    done



done



echo "" >> "$REPORT"

echo "============================================================" >> "$REPORT"
echo "TOTAL DATASETS:" >> "$REPORT"
echo "$TOTAL" >> "$REPORT"
echo "============================================================" >> "$REPORT"


cat "$REPORT"


