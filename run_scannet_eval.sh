#!/bin/bash
set -euo pipefail

GT_DIR="${GT_DIR:-/path/to/scannet/gt}"
VIDEO_DIR="${VIDEO_DIR:-/path/to/scannet/scannet_val}"
SAM1_CKPT="${SAM1_CKPT:-/path/to/sam_vit_h_4b8939.pth}"
SAM2_CKPT="${SAM2_CKPT:-/path/to/sam2.1_hiera_large.pt}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
GPU_ID="${GPU_ID:-0}"

echo "=========================================================="
echo "Starting Savvy ScanNet evaluation"
echo "=========================================================="

python -m evaluation.savvy_scannet_runner \
        --gt_dir "$GT_DIR" \
        --video_dir "$VIDEO_DIR" \
        --sam1_ckpt "$SAM1_CKPT" \
        --sam2_ckpt "$SAM2_CKPT" \
        --sam2_cfg "$SAM2_CFG" \
        --eval_dir "./scannet_eval/" \
        --vis_dir "./scannet_eval/vis" \
        --gpu "$GPU_ID" \
        --sam_points 32 \
        --segmenter_stride 3 \
        --k_keep 0 \
        --buffer_size 30 \
        --margin 0.1
