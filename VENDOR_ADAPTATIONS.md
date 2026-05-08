# Adapted SAM/SAM2 Subsets

This repository includes minimal runtime subsets of SAM1 (`segment_anything/`) and
SAM2 (`sam2/`) so the `savvy/` package can run without a full vendor checkout.

## Adapted Files

SAM2:
- `sam2/sam2_video_predictor.py`
- `sam2/modeling/sam2_base.py`
- `sam2/utils/misc.py`

SAM1:
- `segment_anything/automatic_mask_generator.py`
- `segment_anything/utils/amg.py`

## Included Runtime Dependencies

The remaining files under `sam2/` and `segment_anything/` are package initializers,
model builders, model components, transforms, or configs required by the adapted
files and current evaluation runners.

The precompiled SAM2 extension is not included. If unavailable, SAM2 falls back by
skipping the small-hole postprocessing step that depends on the extension.

Removed vendor files include demos, benchmarks, unused predictors, original backup
copies, training-only configs, notebook checkpoints, generated caches, and the unused
SAM2 image/automatic-mask-generator path.
