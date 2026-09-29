#!/bin/bash

set -e


BASE="/mnt/hdd16T/ToqeerHomeBackup/mujoco_project"

cd "$BASE"


echo "============================================================"
echo " FULL HUMAN FALL DATASET CAMPAIGN"
echo "============================================================"



# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------


SCENARIOS=(
41
42
43
44
)


AGES=(
55
65
75
)


SEXES=(
male
female
)


HEIGHT=1.75



MASTER="outputs/master_dataset_index.csv"



mkdir -p outputs/campaign_logs


echo \
"scenario,age,sex,height,weight,dataset_path,frames,video_frames,validation" \
> "$MASTER"



# ------------------------------------------------------------
# RUN ALL EXPERIMENTS
# ------------------------------------------------------------


for SCENARIO in "${SCENARIOS[@]}"
do

for AGE in "${AGES[@]}"
do


for SEX in "${SEXES[@]}"
do



echo
echo "============================================================"
echo "RUNNING"
echo "Scenario=$SCENARIO"
echo "Age=$AGE"
echo "Sex=$SEX"
echo "============================================================"



LOG="outputs/campaign_logs/scenario${SCENARIO}_age${AGE}_${SEX}.log"



python fall_dispatcher.py > "$LOG" 2>&1 <<EOF2
$SCENARIO
$AGE
$HEIGHT
$SEX
78
EOF2




LATEST=$(ls -td outputs/scenario${SCENARIO}_* | head -1)



echo "OUTPUT:"
echo "$LATEST"



# ------------------------------------------------------------
# CREATE DATASET CONTAINER
# ------------------------------------------------------------


DATASET="outputs/scenario${SCENARIO}_dataset"


mkdir -p "$DATASET"



FINAL="$DATASET/$(basename "$LATEST")"



mkdir -p "$FINAL"


cp -r "$LATEST"/* "$FINAL"/



# ------------------------------------------------------------
# FRAME EXTRACTION
# ------------------------------------------------------------


mkdir -p "$FINAL/frames"


python tools/extract_video_frames.py \
"$FINAL/scenario${SCENARIO}_native_render.mp4" \
"$FINAL/frames" \
|| true




# ------------------------------------------------------------
# ORGANIZATION
# ------------------------------------------------------------


bash organize_all_fall_datasets.sh




# ------------------------------------------------------------
# COUNT FRAMES
# ------------------------------------------------------------


FRAMES=$(find "$FINAL/frames" -name "*.png" | wc -l)



VIDEO_FRAMES=$(ffprobe \
-v error \
-select_streams v:0 \
-show_entries stream=nb_frames \
-of csv=p=0 \
"$FINAL/scenario${SCENARIO}_native_render.mp4" \
2>/dev/null || echo 0)



VALIDATION=$(grep -m1 "score" "$FINAL"/*validation* 2>/dev/null || echo "NA")




echo \
"$SCENARIO,$AGE,$SEX,$HEIGHT,auto,$FINAL,$FRAMES,$VIDEO_FRAMES,\"$VALIDATION\"" \
>> "$MASTER"



done
done
done



echo
echo "============================================================"
echo " CAMPAIGN COMPLETE"
echo "============================================================"



echo
echo "MASTER INDEX:"
echo "$MASTER"



