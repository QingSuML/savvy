#!/bin/bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/hm3d}"
SAM1_CKPT="${SAM1_CKPT:-/path/to/sam_vit_h_4b8939.pth}"
SAM2_CKPT="${SAM2_CKPT:-/path/to/sam2.1_hiera_large.pt}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_l.yaml}"
GPU_ID="${GPU_ID:-0}"

echo "=========================================================="
echo "Starting Savvy HM3D evaluation"
echo "=========================================================="

python -m evaluation.savvy_hm3d_runner \
        --data_root "$DATA_ROOT" \
        --sam1_ckpt "$SAM1_CKPT" \
        --sam2_ckpt "$SAM2_CKPT" \
        --sam2_cfg "$SAM2_CFG" \
        --eval_dir "./hm3d_eval/" \
        --vis_dir "./hm3d_eval/vis" \
        --gpu "$GPU_ID" \
        --sam_points 32 \
        --segmenter_stride 3 \
        --k_keep 0 \
        --buffer_size 30
