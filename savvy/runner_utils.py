from sam2.build_sam import build_sam2_video_predictor
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def setup_models(
    device,
    sam1_ckpt_path,
    sam2_ckpt_path,
    sam2_model_cfg,
    min_mask_region_area=100,
):
    """Initialize SAM1 and SAM2 models once for a runner process."""
    print(f"Initializing models on {device}...")
    sam1 = sam_model_registry["vit_h"](checkpoint=sam1_ckpt_path).to(device)
    sam1_mask_generator = SamAutomaticMaskGenerator(
        model=sam1,
        pred_iou_thresh=0.88,
        box_nms_thresh=0.7,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        min_mask_region_area=min_mask_region_area,
    )

    predictor = build_sam2_video_predictor(
        sam2_model_cfg,
        sam2_ckpt_path,
        device=device,
    )
    predictor.num_maskmem = 7
    predictor.memory_temporal_stride_for_eval = 1
    predictor.use_mask_input_as_output_without_sam = True
    return sam1_mask_generator, predictor


def choose_frame_sampling(num_frames, max_frames=1500):
    """Return the sample interval and frame cutoff used by the dataset runners."""
    if num_frames <= 300:
        return 1, num_frames
    if num_frames <= 1000:
        return 2, num_frames
    if num_frames <= max_frames:
        return 3, num_frames
    return 3, max_frames
