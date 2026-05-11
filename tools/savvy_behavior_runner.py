import os
import glob
import shutil
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from savvy import Savvy
from savvy.runner_utils import choose_frame_sampling, setup_models
from utils import show_masks_fast
from evaluation.oga_metrics import ScannetVOSDataset, VOSEvaluator

def build_ablation_tag(cfg, args):
    tags = []
    if not cfg.get("use_appearance_self_consistency", True):
        tags.append("no_selfcons")
    if not cfg.get("use_seniority_suppression", True):
        tags.append("no_seniority")
    if not cfg.get("use_transient_handshake", True):
        tags.append("no_handshake")
    if not cfg.get("use_hierarchical_merge", True):
        tags.append("no_hmerge")
    if not cfg.get("use_part_whole_absorption", True):
        tags.append("no_absorb")
    if cfg.get("buffer_size", 30) != 30:
        tags.append("buffer_" + str(cfg.get("buffer_size", 30)))
    tags.append("Savvy")
    if args.exp_tag:
        tags.append(args.exp_tag)
    return "full" if not tags else "__".join(tags)

def process_single_scene(args, scene_id, gt_base_dir, video_base_dir, out_eval_dir, out_vis_dir,
                         sam1_mask_generator, predictor, save_visualizations=True,
                         buffer_size=30, sam_num_points_per_side=32, segmenter_stride=3,
                         adapt_num_points=False, ablation_config=None,
                         min_promotion_area=-1,
                         min_promotion_area_ratio=-1.0,
                         behavior_log_dir=None,
                        ):
    """Runs the Savvy pipeline on a single scene and saves the raw masks."""
    video_dir = os.path.join(video_base_dir, scene_id)
    gt_scene_dir = os.path.join(gt_base_dir, scene_id, "instance")

    if not os.path.exists(video_dir) or not os.path.exists(gt_scene_dir):
        print(f"Skipping {scene_id}: Missing video or GT data.")
        return False

    num_gt_frames = len(glob.glob(os.path.join(gt_scene_dir, "*.png")))
    interval, cutoff = choose_frame_sampling(num_gt_frames)

    print(f"\n--- Processing {scene_id} ---")
    print(f"Frames: {num_gt_frames} -> Capped at {cutoff}, Interval: {interval}")

    # Initialize state (ensuring we handle the inference_state reset correctly)
    inference_state = predictor.init_state(
        video_path=video_dir, start_frame_idx=0, end_frame_idx=cutoff,
        sample_interval=interval, offload_video_to_cpu=True, offload_state_to_cpu=True
    )

    behavior_log_path = None
    if behavior_log_dir is not None:
        behavior_log_path = os.path.join(behavior_log_dir, scene_id, "savvy_behavior_stats.csv")

    video_segmenter = Savvy(
        predictor,
        inference_state,
        segmenter=sam1_mask_generator,
        buffer_size=buffer_size,
        iou_threshold=0.5,
        survival_threshold=0.1,
        margin=args.margin, # seniority suppresssion. NOT border delay
        memory_strength=0.7,
        sam_num_points_per_side=sam_num_points_per_side,
        segmenter_stride=segmenter_stride,
        adapt_num_points=adapt_num_points,
        handshake_min_agreement_hits=args.handshake_min_agreement_hits,
        handshake_min_visible_frames_for_promotion=args.handshake_min_visible_frames_for_promotion,
        handshake_min_max_area_for_promotion=args.handshake_min_max_area_for_promotion,
        handshake_min_max_area_ratio_for_promotion=args.handshake_min_max_area_ratio_for_promotion,
        behavior_log_path=behavior_log_path,
    )

    merged_results = video_segmenter.track(external_masks_per_frame=None, start_frame_idx=0)

    os.makedirs(out_eval_dir, exist_ok=True)

    if save_visualizations:
        os.makedirs(out_vis_dir, exist_ok=True)

    frame_names = sorted([f for f in os.listdir(video_dir) if f.endswith((".jpg", ".jpeg"))],
                         key=lambda p: int(''.join(filter(str.isdigit, os.path.splitext(p)[0]))))

    for out_frame_idx, (frame_key, frame_masks) in enumerate(tqdm(merged_results.items(), desc=f"Saving {scene_id}")):
        img_path = os.path.join(video_dir, frame_names[out_frame_idx * interval])
        img = np.array(Image.open(img_path))
        h, w = img.shape[:2]

        label_map = np.zeros((h, w), dtype=np.uint16)
        if frame_masks:
            for obj_id, mask in sorted(
                frame_masks.items(),
                key=lambda kv: int(np.squeeze(kv[1]).sum()),
                reverse=True,
            ):
                label_map[mask.reshape(h, w).astype(bool)] = int(obj_id) + 1

        Image.fromarray(label_map).save(os.path.join(out_eval_dir, f"{frame_key * interval:05d}.png"))

        if save_visualizations:
            fig, ax = plt.subplots(figsize=(8, 5))
            show_masks_fast(frame_masks, ax, show_id=True, img=img, seed=13)
            plt.savefig(os.path.join(out_vis_dir, f"frame_{frame_key * interval:05d}.png"),
                        bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig)

    with open(os.path.join(out_eval_dir, ".completed"), "w") as f: f.write("done")
    return True

def main(args):
    # --- 1. Setup ---
    unsupported_ablation_flags = [
        name for name in (
            "no_hierarchical_merge",
            "no_transient_handshake",
            "no_appearance_self_consistency",
            "no_seniority_suppression",
            "no_part_whole_absorption",
            "no_track_consolidation_hysteresis",
        )
        if getattr(args, name, False)
    ]
    if unsupported_ablation_flags:
        raise ValueError(
            "The cleaned Savvy runner does not support disabling pipeline "
            f"components via {', '.join('--' + name for name in unsupported_ablation_flags)}."
        )

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    sam1_gen, predictor = setup_models(device, args.sam1_ckpt, args.sam2_ckpt, args.sam2_cfg)

    ablation_config = {
        "buffer_size": args.buffer_size,
    }
    ablation_tag = build_ablation_tag(ablation_config, args)
    eval_root = os.path.join(args.eval_dir, ablation_tag)
    vis_root = os.path.join(args.vis_dir, ablation_tag)
    behavior_log_root = os.path.join(args.behavior_log_dir, ablation_tag)
    os.makedirs(eval_root, exist_ok=True)
    os.makedirs(vis_root, exist_ok=True)
    os.makedirs(behavior_log_root, exist_ok=True)

    # --- 2. History & Folder Mapping ---
    dataset = ScannetVOSDataset(gt_dir=args.gt_dir, pred_dir=eval_root)
    evaluator = VOSEvaluator(dataset, max_pattern_size=3)
    filename = "oga_full_results.json" if args.exp_tag is None else f"oga_full_results_{args.exp_tag}.json"
    history_file = os.path.join(eval_root, filename)

    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                evaluator.results_per_scene.update(json.load(f))
            shutil.copy(history_file, history_file + ".bak")
            print(f"[*] History merged: {len(evaluator.results_per_scene)} scenes recovered.")
        except Exception as e: print(f"[!] Error loading history: {e}")

    # Use scene_id[-12:] to find folders even if they are renamed (TOP1_sceneXXXX_XX)
    def get_disk_map(root):
        if not os.path.exists(root): return {}
        return {d[-12:]: d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}

    eval_disk_map = get_disk_map(eval_root)
    vis_disk_map = get_disk_map(vis_root)

    all_scenes = args.scene_names if args.scene_names else sorted([s for s in dataset.get_scene_ids() if s.endswith("_00")])
    if args.resume_from and args.resume_from in all_scenes:
        all_scenes = all_scenes[all_scenes.index(args.resume_from):]

    # --- 3. Master Loop ---
    for scene_id in all_scenes:
        if scene_id in evaluator.results_per_scene and not args.force_rerun:
            print(f"[*] Skipping {scene_id} (found in JSON record).")
            continue

        actual_eval_folder = eval_disk_map.get(scene_id, scene_id)
        actual_vis_folder = vis_disk_map.get(scene_id, scene_id)
        scene_eval_dir = os.path.join(eval_root, actual_eval_folder)
        scene_vis_dir = os.path.join(vis_root, actual_vis_folder)

        if os.path.exists(os.path.join(scene_eval_dir, ".completed")) and not args.force_rerun:
            print(f"[*] Skipping {scene_id} (folder complete). Syncing metrics...")
            if args.exp_tag is not None:
                if args.exp_tag.split('_')[0] in ["clutter"]:
                    evaluator.evaluate_clutter_test(scene_names=scene_id)
                elif args.exp_tag.split('_')[0] in ["flicker"]:
                    evaluator.evaluate_flickering_test(scene_names=scene_id)
                elif args.exp_tag.split('_')[0] in ["dropout"]:
                    evaluator.evaluate_drop_test(scene_names=scene_id)
                elif args.exp_tag.split('_')[0] in ["sever"]:
                    evaluator.evaluate_macro_sever_test(scene_names=scene_id)
                elif args.exp_tag.split('_')[0] in ["void", "dilation"]:
                    evaluator.evaluate_dilation_test(scene_names=scene_id)
                else:
                    evaluator.evaluate(scene_names=scene_id)
            else:
                evaluator.evaluate(scene_names=scene_id)
            evaluator.save_results(output_dir=eval_root, tag=args.exp_tag)
            continue

        if os.path.exists(scene_eval_dir):
            shutil.rmtree(scene_eval_dir, ignore_errors=True)
            shutil.rmtree(scene_vis_dir, ignore_errors=True)


        if process_single_scene(args, scene_id, args.gt_dir, args.video_dir, scene_eval_dir, scene_vis_dir,
                               sam1_gen, predictor, save_visualizations=not args.no_vis,
                               buffer_size=args.buffer_size, sam_num_points_per_side=args.sam_points,
                               segmenter_stride=args.segmenter_stride,
                               adapt_num_points=args.adapt_num_points,
                               ablation_config=ablation_config,
                               min_promotion_area=args.min_promotion_area,
                               min_promotion_area_ratio=args.min_promotion_area_ratio,
                               behavior_log_dir=behavior_log_root,
                               ):
            evaluator.evaluate(scene_names=scene_id)
            evaluator.save_results(output_dir=eval_root, tag=args.exp_tag)

    # --- 4. SELECTIVE PRUNING & RENAMING ---
    if evaluator.results_per_scene and not args.scene_names and args.k_keep > 0:
        print(f"\n[*] Cleaning up and organizing Top/Bottom {args.k_keep} for both IP and VPQ_inf...")

        # 1. Rank by IP (Combined)
        ranked_ip = sorted(
            evaluator.results_per_scene.keys(),
            key=lambda sid: evaluator.results_per_scene[sid].get('identity_persistence', {}).get('combined', 0),
            reverse=True
        )
        top_ip = ranked_ip[:args.k_keep]
        bot_ip = ranked_ip[-args.k_keep:]

        # 2. Rank by VPQ_inf
        ranked_vpq = sorted(
            evaluator.results_per_scene.keys(),
            key=lambda sid: evaluator.results_per_scene[sid].get('baselines', {}).get('VPQ_inf', 0),
            reverse=True
        )
        top_vpq = ranked_vpq[:args.k_keep]
        bot_vpq = ranked_vpq[-args.k_keep:]

        # 3. Combine into a master keep set (removes duplicates automatically)
        keep_set = set(top_ip + bot_ip + top_vpq + bot_vpq)

        for scene_id in list(evaluator.results_per_scene.keys()):
            old_eval_path = os.path.join(eval_root, scene_id)
            old_vis_path = os.path.join(vis_root, scene_id)

            if scene_id not in keep_set:
                # Remove the "average" middle-tier performers
                shutil.rmtree(old_eval_path, ignore_errors=True)
                shutil.rmtree(old_vis_path, ignore_errors=True)
            else:
                # Build a dynamic prefix depending on which lists the scene made it into
                tags = []
                if scene_id in top_ip:  tags.append(f"TopIP{top_ip.index(scene_id) + 1}")
                if scene_id in bot_ip:  tags.append(f"BotIP{list(reversed(bot_ip)).index(scene_id) + 1}")
                if scene_id in top_vpq: tags.append(f"TopVPQ{top_vpq.index(scene_id) + 1}")
                if scene_id in bot_vpq: tags.append(f"BotVPQ{list(reversed(bot_vpq)).index(scene_id) + 1}")

                prefix = "_".join(tags) + "_"

                # New Folder Names
                new_eval_path = os.path.join(eval_root, f"{prefix}{scene_id}")
                new_vis_path = os.path.join(vis_root, f"{prefix}{scene_id}")

                # Rename the folders
                if os.path.exists(old_eval_path):
                    os.rename(old_eval_path, new_eval_path)
                if os.path.exists(old_vis_path):
                    os.rename(old_vis_path, new_vis_path)

                print(f"[*] Organized: {prefix}{scene_id}")

    print(f"\n[*] Cleanup complete. PNGs kept for the best and worst performers across IP and VPQ.")
    evaluator.print_summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default="/path/to/scannet/gt")
    parser.add_argument("--video_dir", type=str, default="/path/to/scannet/scannet_val")
    parser.add_argument("--eval_dir", type=str, default="./eval_predictions")
    parser.add_argument("--vis_dir", type=str, default="./eval_visualizations")
    parser.add_argument("--behavior_log_dir", type=str, default="./savvy_behavior_logs")
    parser.add_argument("--sam1_ckpt", type=str, default="/path/to/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam2_ckpt", type=str, default="/path/to/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--buffer_size", type=int, default=30)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--sam_points", type=int, default=32)
    parser.add_argument("--segmenter_stride", type=int, default=3)
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--scene_names", type=str, nargs='+', default=None)
    parser.add_argument("--k_keep", type=int, default=10)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--adapt_num_points", action='store_true')

    parser.add_argument("--no_transient_handshake", action="store_true")
    parser.add_argument("--no_appearance_self_consistency", action="store_true")
    parser.add_argument("--no_seniority_suppression", action="store_true")
    parser.add_argument("--no_part_whole_absorption", action="store_true")
    parser.add_argument("--no_hierarchical_merge", action="store_true")

    parser.add_argument("--min_promotion_area", type=int, default=-1)
    parser.add_argument("--min_promotion_area_ratio", type=float, default=-1.0)

    parser.add_argument("--handshake_min_agreement_hits", type=int, default=5)
    parser.add_argument("--handshake_min_visible_frames_for_promotion", type=int, default=3)
    parser.add_argument("--handshake_min_max_area_for_promotion", type=int, default=-1)
    parser.add_argument("--handshake_min_max_area_ratio_for_promotion", type=float, default=-1.0)

    parser.add_argument("--no_track_consolidation_hysteresis", action="store_true")
    parser.add_argument("--consolidation_min_win_streak", type=int, default=3)
    parser.add_argument("--consolidation_state_ttl", type=int, default=10)

    parser.add_argument("--exp_tag", type=str, default=None)

    main(parser.parse_args())
