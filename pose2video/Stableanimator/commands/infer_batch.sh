#!/bin/bash

# Define variables with default values
DATASET_PATH=${1:-""}
OUTPUT_PATH=${2:-""}
SAVE_STAGE1_PATH=${3:-""}
DEVICES=${4:-"0,1,2,3,4,5,6,7"}
CFG=${5:-"3.5"}


# Check if required arguments are provided
if [ -z "$DATASET_PATH" ] || [ -z "$OUTPUT_PATH" ] || [ -z "$SAVE_STAGE1_PATH" ]; then
    echo "Usage: $0 <dataset_path> <output_path> <save_stage1_path> [devices] [cfg]"
    exit 1
fi

echo "--- 1. Running ready_mp4_path_ref_img.py ---"
python scripts/ready_mp4_path_ref_img.py \
    --dataset_path "$DATASET_PATH" \
    --output_path "$OUTPUT_PATH"

echo "--- 2. Running sample_posetoready_all.py ---"
python scripts/sample_posetoready_all.py \
    --dataset_path "$DATASET_PATH" \
    --save_stage1_path "$SAVE_STAGE1_PATH" \
    --output_path "$OUTPUT_PATH" \
    --cfg "$CFG" \
    --devices "$DEVICES"

echo "--- 3. Running infer.py ---"
python scripts/infer.py \
    --raw_path "$OUTPUT_PATH/results" \
    --devices "$DEVICES"

echo "--- All scripts finished successfully. ---"