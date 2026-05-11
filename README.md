# Savvy

Savvy is an open-world video segmentation pipeline for tracking segmentation
identities across video frames. This repository also includes the accompanying
Open-world Granularity-Agnostic (OGA) evaluation code, benchmark runners, and
figure-generation utilities.

## Demo Videos

See [qualitative results](QUALITATIVE_RESULTS.md) for the full demo gallery.

Click any thumbnail to watch the demo on YouTube.

| Demo | Demo | Demo |
| --- | --- | --- |
| [![Savvy demo video 1](https://img.youtube.com/vi/ZF5Q4DqltdQ/hqdefault.jpg)](https://youtu.be/ZF5Q4DqltdQ) | [![Savvy demo video 2](https://img.youtube.com/vi/GbHqHBUKl3w/hqdefault.jpg)](https://youtu.be/GbHqHBUKl3w) | [![Savvy demo video 3](https://img.youtube.com/vi/g7WfW6JBB0o/hqdefault.jpg)](https://youtu.be/g7WfW6JBB0o) |
| [![Savvy demo video 4](https://img.youtube.com/vi/4kBFNuYYHdU/hqdefault.jpg)](https://youtu.be/4kBFNuYYHdU) | [![Savvy demo video 5](https://img.youtube.com/vi/7Kv2Brfmqes/hqdefault.jpg)](https://youtu.be/7Kv2Brfmqes) |  |

## Repository Layout

- `savvy/`: core Savvy pipeline and tracking components.
- `inference/`: generic Savvy inference for a directory of video frames.
- `evaluation/`: ScanNet, HM3D, and VIPSeg runners plus shared OGA metrics.
- `visualization/`: cleaned source modules for support matrices, behavior
  matrices, sequence strips, and identity-event curves.
- `tools/`: runtime and memory profiling utilities for ScanNet validation runs.
- `notebooks/`: lightweight example notebooks that import the cleaned modules.
- `sam2/` and `segment_anything/`: minimal adapted SAM2/SAM1 runtime subsets.
- `run_*.sh`: root-level launch scripts for common inference and evaluation jobs.

See `VENDOR_ADAPTATIONS.md` for the adapted SAM/SAM2 file list.

## Setup

This code expects a Python environment with PyTorch/CUDA support and the common
scientific Python stack used by SAM/SAM2 workflows:

- `torch`, `torchvision`
- `numpy`, `scipy`, `pandas`
- `opencv-python`
- `Pillow`
- `matplotlib`
- `hydra-core`, `omegaconf`

SAM1 and SAM2 checkpoints are required. The launch scripts expose these as
`SAM1_CKPT`, `SAM2_CKPT`, and `SAM2_CFG`; set those paths for the local environment.

## Generic Inference

Run Savvy on a directory containing an ordered sequence of frames:

```bash
./run_savvy_inference.sh /path/to/frames /path/to/output 0
```

Outputs:

- `masks/`: `uint16` PNG instance maps.
- `visualizations/`: optional overlay images.
- `metadata.json`: frame list and inference settings.

For additional options:

```bash
python -m inference.savvy_video_inference --help
```

## Runtime Profiling

We provide a lightweight profiling script for the submitted Savvy inference
pipeline on the ScanNet validation split.

The script runs the default full Savvy configuration, including hierarchical
mask discovery, SAM2 propagation, deferred admission, track consolidation,
memory pruning, and active-track control.

Timing is reported as cumulative average FPS:

```text
processed frames / elapsed inference time
```

Visualization and metric computation are excluded from timing.

Example command:

```bash
python tools/profile_savvy_runtime.py \
    --device cuda:0 \
    --output runtime_logs/scannet_val/ \
    --gt_dir /path/to/scannet/gt \
    --video_dir /path/to/scannet/scannet_val \
    --sam1_ckpt /path/to/sam_vit_h_4b8939.pth \
    --sam2_ckpt /path/to/sam2.1_hiera_large.pt
```

The generated plots summarize runtime behavior over all ScanNet validation
scenes. Thin curves show individual scenes and the dark curve shows the mean.
For readability, overlay traces are y-axis capped at the 95th percentile; the
raw data and mean curve are unchanged.

![Savvy runtime profile over ScanNet validation scenes](tools/runtime_profile_all.png)

On an NVIDIA A6000, the final mean cumulative throughput is 7.4 FPS for the
full Savvy pipeline. GPU memory remains bounded at roughly 6-7 GB while the
object set grows over time, and the number of active masks per frame remains
controlled throughout inference.

## Benchmark Evaluation

The root scripts are the main entry points:

```bash
./run_scannet_eval.sh
./run_hm3d_eval.sh
./run_vipseg_eval.sh 0
```

Before running, edit dataset and checkpoint paths in the scripts.

The shared OGA metric implementation lives in `evaluation/oga_metrics/`.
ScanNet/HM3D evaluation outputs are written as `oga_full_results*.json/csv`.
VIPSeg evaluation outputs are written as `vipseg_savvy_oga_*`.

## Visualization

Reusable visualization code lives under `visualization/`:

- `support_matrix_behavior_matrix.py`
- `identity_discovery_reassociation_events.py`
- `sequence_strips.py`

A lightweight notebook with placeholder paths is available at:

```text
notebooks/visualization_examples.ipynb
```

The notebook demonstrates sequence strips, support matrices, OGA behavior
matrices, and reassociation-event curves.
