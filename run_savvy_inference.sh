#!/bin/bash
set -euo pipefail

VIDEO_DIR=${1:?Usage: ./run_savvy_inference.sh /path/to/frames /path/to/output [gpu_id]}
OUTPUT_DIR=${2:?Usage: ./run_savvy_inference.sh /path/to/frames /path/to/output [gpu_id]}
GPU_ID=${3:-0}

python -m inference.savvy_video_inference \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --gpu "$GPU_ID"
