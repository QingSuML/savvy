# Inference

Generic Savvy inference for a directory of video frames.

Example:

```bash
./run_savvy_inference.sh /path/to/frames /path/to/output 0
```

Outputs:
- `masks/`: `uint16` PNG instance maps, where `0` is background and object labels are `obj_id + 1`.
- `visualizations/`: optional mask overlays, omitted when `--no_vis` is passed to `python -m inference.savvy_video_inference`.
- `metadata.json`: frame list and inference settings.
