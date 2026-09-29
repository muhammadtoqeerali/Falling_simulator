#!/usr/bin/env bash

VIDEOS="
outputs/scenario23_age25_h1p62_sex_female_w58p0_20260824_083429/fall_scenario23_age25_20260824_083833_highrate_truth_display.mp4
outputs/scenario25_age25_h1p62_sex_female_w58p0_20260824_083843/fall_scenario25_age25_20260824_084000_highrate_truth_display.mp4
outputs/scenario39_age25_h1p62_sex_female_w58p0_20260824_084007/fall_forward_height_age25_20260824_084213_highrate_truth_display.mp4
outputs/scenario43_age25_h1p62_sex_female_w58p0_20260824_084238/fall_forward_ladder_age25_20260824_084402_highrate_truth_display.mp4
"

echo "============================================================"
echo "VERIFYING FOUR PAPER-SCENARIO VIDEOS"
echo "============================================================"

good=0

echo "$VIDEOS" | while IFS= read -r f
do
    [ -n "$f" ] || continue

    echo
    echo "------------------------------------------------------------"
    echo "$f"
    echo "------------------------------------------------------------"

    if [ ! -f "$f" ]; then
        echo "MISSING"
        continue
    fi

    ffprobe \
        -v error \
        -select_streams v:0 \
        -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration,size \
        -of default=noprint_wrappers=1 \
        "$f"

    echo "Decode test:"

    if ffmpeg \
        -v error \
        -i "$f" \
        -map 0:v:0 \
        -f null - \
        </dev/null
    then
        echo "PASS"
    else
        echo "FAIL"
    fi
done
