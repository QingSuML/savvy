import os
import time
import json
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from savvy import Savvy
from savvy.runner_utils import setup_models

def id_to_rgb(obj_id):
    """Converts an integer tracking ID to an RGB color for panoptic encoding."""
    r = obj_id % 256
    g = (obj_id // 256) % 256
    b = (obj_id // 65536) % 256
    return [r, g, b]

def process_and_save_vipseg_format(scene_id, merged_results, frame_names, video_dir, output_dir):
    """Converts Savvy output dict to VIPSeg RGB panoptic PNGs and COCO-style info."""
    annotations = []

    for out_frame_idx, frame_name in enumerate(frame_names):
        frame_masks = merged_results.get(out_frame_idx, {})

        # Use the original image size for rasterization.
        img_path = os.path.join(video_dir, frame_name)
        with Image.open(img_path) as img:
            h, w = img.size[1], img.size[0]

        # Flatten masks onto a 2D map using area-priority rasterization.
        # Large masks are written first, small masks later, so small masks
        # are not swallowed by large ones in overlap regions.
        id_map = np.full((h, w), fill_value=-1, dtype=np.int32)

        sorted_masks = []
        for obj_id, mask in frame_masks.items():
            mask_bool = mask.reshape(h, w).astype(bool)
            area = int(mask_bool.sum())
            if area > 0:
                sorted_masks.append((obj_id, mask_bool, area))

        sorted_masks.sort(key=lambda x: x[2], reverse=True)

        for obj_id, mask_bool, _ in sorted_masks:
            id_map[mask_bool] = int(obj_id) + 1

        pan_format = np.zeros((h, w, 3), dtype=np.uint8)
        segments_info = []

        # Extract strictly visible flattened regions and bboxes.
        unique_ids = np.unique(id_map)
        for obj_id_int in unique_ids:
            if obj_id_int == -1:
                continue

            mask_bool = (id_map == obj_id_int)
            area = int(mask_bool.sum())
            if area == 0:
                continue

            color = id_to_rgb(obj_id_int)
            pan_id = color[0] + color[1] * 256 + color[2] * 256 * 256

            pan_format[mask_bool] = color

            y_indices, x_indices = np.where(mask_bool)
            x_min = int(x_indices.min())
            y_min = int(y_indices.min())
            width = int(x_indices.max() - x_min)
            height = int(y_indices.max() - y_min)

            segments_info.append({
                "id": int(pan_id),
                "category_id": 1,
                "isthing": 1,
                "iscrowd": 0,
                "bbox": [x_min, y_min, width, height],
                "area": int(area)
            })

        pan_pred_dir = os.path.join(output_dir, 'pan_pred', scene_id)
        os.makedirs(pan_pred_dir, exist_ok=True)
        out_img_name = frame_name.split('/')[-1].split('.')[0] + '.png'
        Image.fromarray(pan_format).save(os.path.join(pan_pred_dir, out_img_name))

        annotations.append({
            "file_name": frame_name.split('/')[-1],
            "segments_info": segments_info
        })

    anno_data = {"video_id": scene_id, "annotations": annotations}
    with open(os.path.join(pan_pred_dir, 'anno.json'), 'w') as f:
        json.dump(anno_data, f)

    return anno_data

def main(args):
    # Runtime setup.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    sam1_gen, predictor = setup_models(
        device,
        args.sam1_ckpt,
        args.sam2_ckpt,
        args.sam2_cfg,
        min_mask_region_area=0,
    )

    # Load VIPSeg metadata.
    with open(args.pan_gt_json_file, 'r') as f:
        data_val_json = json.load(f)

    predictions = []
    output_dir = os.path.join(args.submit_dir)
    os.makedirs(output_dir, exist_ok=True)

    root_path = args.video_dir
    processing_order = range(len(data_val_json['videos']))

    # Main video loop.
    for video_idx in tqdm(processing_order, desc="Evaluating Savvy on VIPSeg"):
        torch.cuda.reset_peak_memory_stats()
        
        video_info = data_val_json['videos'][video_idx]
        scene_id = video_info['video_id']
        video_dir = os.path.join(root_path, scene_id)

        if not os.path.exists(video_dir):
            print(f"Skipping {scene_id}: Missing video data at {video_dir}")
            continue

        pan_pred_dir = os.path.join(output_dir, 'pan_pred', scene_id)
        anno_file = os.path.join(pan_pred_dir, 'anno.json')
        
        if os.path.exists(anno_file):
            try:
                with open(anno_file, 'r') as f:
                    predictions.append(json.load(f))
                print(f"[*] Skipping {scene_id} - already processed.")
                continue
            except json.JSONDecodeError:
                print(f"[!] Corrupted JSON found for {scene_id}. Reprocessing...")

        frame_names = [
            p for p in sorted(os.listdir(video_dir))
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
        
        # Initialize SAM2 state.
        inference_state = predictor.init_state(
            video_path=video_dir, start_frame_idx=0, end_frame_idx=len(frame_names),
            sample_interval=1, offload_video_to_cpu=True, offload_state_to_cpu=True
        )

        # Initialize Savvy.
        video_segmenter = Savvy(
            predictor, inference_state, segmenter=sam1_gen, 
            buffer_size=args.buffer_size, iou_threshold=0.5, survival_threshold=0.1,
            margin=0.1, memory_strength=0.5, sam_num_points_per_side=args.sam_points,
            segmenter_stride=args.segmenter_stride,
        )

        # Run tracking.
        merged_results = video_segmenter.track(external_masks_per_frame=None, start_frame_idx=0)

        # Format and save in the VIPSeg submission structure.
        anno = process_and_save_vipseg_format(scene_id, merged_results, frame_names, video_dir, output_dir)
        predictions.append(anno)

        # Release per-video tensors before the next video.
        del inference_state
        torch.cuda.empty_cache()

    # Save the master prediction JSON.
    file_path = os.path.join(output_dir, 'pred.json')
    with open(file_path, 'w') as f:
        json.dump({'annotations': predictions}, f)
        
    print(f"\n[*] Inference complete. Formatted data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Savvy inference on VIPSeg for class-agnostic eval.")
    parser.add_argument('--submit_dir', type=str, default='./savvy_vipseg_output', help="Where to save pred.json and pan_pred/ PNGs")
    parser.add_argument('--video_dir', type=str, default='/path/to/VIPSeg_720P/images', help="Path to VIPSeg images")
    parser.add_argument('--pan_gt_json_file', type=str, default='/path/to/VIPSeg_720P/panoptic_gt_VIPSeg_val.json', help="Path to VIPSeg val json")
    
    # Savvy model arguments.
    parser.add_argument("--sam1_ckpt", type=str, default="/path/to/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam2_ckpt", type=str, default="/path/to/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--buffer_size", type=int, default=10)
    parser.add_argument("--sam_points", type=int, default=64)
    parser.add_argument("--segmenter_stride", type=int, default=3)
    
    main(parser.parse_args())
