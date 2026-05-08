import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from savvy import Savvy
from savvy.runner_utils import setup_models


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def list_frame_names(video_dir):
    frame_names = [
        name for name in os.listdir(video_dir)
        if name.endswith(IMAGE_EXTENSIONS)
    ]
    return sorted(frame_names)


def save_label_map(frame_masks, image_shape, output_path):
    h, w = image_shape
    label_map = np.zeros((h, w), dtype=np.uint16)

    if frame_masks:
        for obj_id, mask in sorted(
            frame_masks.items(),
            key=lambda kv: int(np.squeeze(kv[1]).sum()),
            reverse=True,
        ):
            label_map[np.squeeze(mask).reshape(h, w).astype(bool)] = int(obj_id) + 1

    Image.fromarray(label_map).save(output_path)


def run_inference(args):
    if not os.path.isdir(args.video_dir):
        raise FileNotFoundError(f"Frame directory not found: {args.video_dir}")

    frame_names = list_frame_names(args.video_dir)
    if not frame_names:
        raise RuntimeError(f"No image frames found in: {args.video_dir}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    sam1_gen, predictor = setup_models(
        device,
        args.sam1_ckpt,
        args.sam2_ckpt,
        args.sam2_cfg,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    mask_dir = os.path.join(args.output_dir, "masks")
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(mask_dir, exist_ok=True)
    if not args.no_vis:
        os.makedirs(vis_dir, exist_ok=True)

    print(f"Running Savvy on {len(frame_names)} frames from {args.video_dir}")
    inference_state = predictor.init_state(
        video_path=args.video_dir,
        start_frame_idx=0,
        end_frame_idx=len(frame_names),
        sample_interval=1,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    video_segmenter = Savvy(
        predictor,
        inference_state,
        segmenter=sam1_gen,
        buffer_size=args.buffer_size,
        iou_threshold=0.5,
        survival_threshold=0.1,
        margin=args.margin,
        memory_strength=args.memory_strength,
        sam_num_points_per_side=args.sam_points,
        segmenter_stride=args.segmenter_stride,
        adapt_num_points=args.adapt_num_points,
    )

    merged_results = video_segmenter.track(
        external_masks_per_frame=None,
        start_frame_idx=0,
    )

    for frame_idx, frame_name in enumerate(tqdm(frame_names, desc="Saving outputs")):
        frame_masks = merged_results.get(frame_idx, {})
        image_path = os.path.join(args.video_dir, frame_name)
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        frame_stem = os.path.splitext(frame_name)[0]

        save_label_map(
            frame_masks,
            image_shape=(h, w),
            output_path=os.path.join(mask_dir, f"{frame_stem}.png"),
        )

        if not args.no_vis:
            import matplotlib.pyplot as plt
            from utils import show_masks_fast

            fig, ax = plt.subplots(figsize=(8, 5))
            show_masks_fast(frame_masks, ax, show_id=True, img=image, seed=13)
            plt.savefig(
                os.path.join(vis_dir, f"{frame_stem}.png"),
                bbox_inches="tight",
                pad_inches=0,
                dpi=100,
            )
            plt.close(fig)

    metadata = {
        "video_dir": args.video_dir,
        "output_dir": args.output_dir,
        "num_frames": len(frame_names),
        "frame_names": frame_names,
        "mask_format": "uint16 PNG, 0=background, object labels are obj_id+1",
        "config": {
            "sam2_cfg": args.sam2_cfg,
            "buffer_size": args.buffer_size,
            "sam_points": args.sam_points,
            "segmenter_stride": args.segmenter_stride,
            "margin": args.margin,
            "memory_strength": args.memory_strength,
            "adapt_num_points": args.adapt_num_points,
        },
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved Savvy outputs to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Savvy on a directory of video frames."
    )
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sam1_ckpt", type=str, default="/path/to/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam2_ckpt", type=str, default="/path/to/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--buffer_size", type=int, default=30)
    parser.add_argument("--sam_points", type=int, default=32)
    parser.add_argument("--segmenter_stride", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--memory_strength", type=float, default=0.7)
    parser.add_argument("--adapt_num_points", action="store_true")
    parser.add_argument("--no_vis", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
