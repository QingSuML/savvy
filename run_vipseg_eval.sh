#!/bin/bash
set -euo pipefail

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Using GPU: $GPU_ID"

SUBMIT_DIR="${SUBMIT_DIR:-./savvy_vipseg_output_64pts}"
IMAGES_DIR="${IMAGES_DIR:-/path/to/VIPSeg_720P/images}"
TRUTH_DIR="${TRUTH_DIR:-/path/to/VIPSeg_720P/panomasksRGB}"
GT_JSON="${GT_JSON:-/path/to/VIPSeg_720P/panoptic_gt_VIPSeg_val.json}"

echo "============================================================"
echo "Step 1: Running Savvy inference on VIPSeg"
echo "============================================================"
python -m evaluation.savvy_vipseg_runner \
    --submit_dir "$SUBMIT_DIR" \
    --video_dir "$IMAGES_DIR" \
    --pan_gt_json_file "$GT_JSON"

echo "============================================================"
echo "Step 2: Running unified Savvy metric evaluation..."
echo "============================================================"
python -m evaluation.savvy_vipseg_evaluation \
    --submit_dir "$SUBMIT_DIR" \
    --truth_dir "$TRUTH_DIR" \
    --pan_gt_json_file "$GT_JSON"

echo "============================================================"
echo "Pipeline Complete! Check $SUBMIT_DIR for the score reports."
echo "============================================================"
