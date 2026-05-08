#!/usr/bin/env python3
"""Qualitative RGB / GT / prediction strip visualizations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_label_png(path):
    return np.asarray(Image.open(path))


def load_rgb_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _numeric_stem(path):
    return int(Path(path).stem)


def sort_paths_numerically(paths):
    return sorted(paths, key=_numeric_stem)


def infer_scannet_sampling_rule(gt_scene_dir):
    """Infer the evaluator's ScanNet GT-to-prediction sampling rule."""
    gt_scene_dir = Path(gt_scene_dir)
    gt_pngs = sort_paths_numerically(gt_scene_dir.glob("*.png"))
    num_gt_frames = len(gt_pngs)

    if num_gt_frames <= 300:
        interval, cutoff = 1, num_gt_frames
    elif num_gt_frames <= 1000:
        interval, cutoff = 2, num_gt_frames
    elif num_gt_frames <= 1500:
        interval, cutoff = 3, num_gt_frames
    else:
        interval, cutoff = 3, 1500

    return interval, cutoff, num_gt_frames


def get_sampled_pngs(pred_dir, num_frames=None, interval=None):
    """Sample PNG paths from a prediction or GT folder."""
    pred_dir = Path(pred_dir)
    pngs = sort_paths_numerically(pred_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {pred_dir}")

    if num_frames is not None and interval is not None:
        raise ValueError("Specify only one of num_frames or interval.")
    if num_frames is not None:
        sampled_indices = np.linspace(0, len(pngs) - 1, num_frames).round().astype(int)
        sampled_indices = np.unique(sampled_indices)
    elif interval is not None:
        sampled_indices = np.arange(0, len(pngs), interval)
    else:
        sampled_indices = np.arange(len(pngs))

    sampled_pngs = [pngs[i] for i in sampled_indices]
    return sampled_pngs, sampled_indices


def get_matched_gt_pngs(gt_dir, pred_sampled_indices, original_interval):
    """Map sampled prediction-sequence indices back to original GT-frame indices."""
    gt_dir = Path(gt_dir)
    gt_pngs = sort_paths_numerically(gt_dir.glob("*.png"))
    if not gt_pngs:
        raise FileNotFoundError(f"No PNG files found in {gt_dir}")

    gt_indices = np.asarray(pred_sampled_indices) * original_interval
    if int(gt_indices.max()) >= len(gt_pngs):
        raise IndexError(f"Mapped GT index {gt_indices.max()} exceeds GT length {len(gt_pngs)}")
    sampled_gt_pngs = [gt_pngs[int(i)] for i in gt_indices]
    return sampled_gt_pngs, gt_indices


def find_frame_dir(scannet_val_dir, scene_name):
    """Try common ScanNet RGB frame locations."""
    scannet_val_dir = Path(scannet_val_dir)
    candidates = [
        scannet_val_dir / scene_name / "color",
        scannet_val_dir / scene_name / "frames" / "color",
        scannet_val_dir / scene_name / "rgb",
        scannet_val_dir / scene_name,
    ]

    for directory in candidates:
        if directory.exists():
            imgs = list(directory.glob("*.jpg")) + list(directory.glob("*.png")) + list(directory.glob("*.jpeg"))
            if imgs:
                return directory
    raise FileNotFoundError(f"Could not find RGB frames for {scene_name} under {scannet_val_dir}")


def get_matched_frame_paths(frame_dir, pred_sampled_indices, original_interval):
    """Map selected prediction-frame indices back to original RGB-frame indices."""
    frame_dir = Path(frame_dir)
    frame_paths = (
        list(frame_dir.glob("*.jpg")) +
        list(frame_dir.glob("*.png")) +
        list(frame_dir.glob("*.jpeg"))
    )
    frame_paths = sort_paths_numerically(frame_paths)
    if not frame_paths:
        raise FileNotFoundError(f"No image files found in {frame_dir}")

    frame_indices = np.asarray(pred_sampled_indices) * original_interval
    if int(frame_indices.max()) >= len(frame_paths):
        raise IndexError(f"Mapped frame index {frame_indices.max()} exceeds RGB frame length {len(frame_paths)}")
    sampled_frame_paths = [frame_paths[int(i)] for i in frame_indices]
    return sampled_frame_paths, frame_indices


def build_global_color_lut(label_paths, seed=0, void_color=(0, 0, 0)):
    """Build one stable color lookup table across sampled label frames."""
    rng = np.random.default_rng(seed)
    ids = set()
    for path in label_paths:
        ids.update(np.unique(load_label_png(path)).tolist())

    lut = {}
    for obj_id in sorted(ids):
        obj_id = int(obj_id)
        if obj_id == 0:
            lut[obj_id] = np.array(void_color, dtype=np.uint8)
        else:
            lut[obj_id] = rng.integers(40, 256, size=3, dtype=np.uint8)
    return lut


def colorize_label(mask, lut):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for obj_id, color in lut.items():
        rgb[mask == obj_id] = color
    return rgb


def _assemble_strip(frames, layout="horizontal"):
    if layout == "horizontal":
        return np.concatenate(frames, axis=1)
    if layout == "vertical":
        return np.concatenate(frames, axis=0)
    raise ValueError("layout must be 'horizontal' or 'vertical'")


def _figure_for_strip(strip, num_frames, layout="horizontal", figsize_scale=4):
    h, w = strip.shape[:2]
    if layout == "horizontal":
        fig_w = figsize_scale * num_frames
        fig_h = fig_w * h / w
    else:
        fig_h = figsize_scale * num_frames
        fig_w = fig_h * w / h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(strip)
    ax.axis("off")
    return fig, ax


def _draw_frame_names(ax, frame_names, single_shape, layout="horizontal"):
    single_h, single_w = single_shape[:2]
    for i, name in enumerate(frame_names):
        x = i * single_w + 5 if layout == "horizontal" else 5
        y = 20 if layout == "horizontal" else i * single_h + 20
        ax.text(
            x,
            y,
            name,
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.5, pad=2),
        )


def visualize_label_strip(
    label_paths,
    seed=0,
    figsize_scale=4,
    show_frame_names=True,
    layout="horizontal",
    save_path=None,
    dpi=200,
):
    """Visualize an explicit list of label-map PNGs as one colorized strip."""
    label_paths = [Path(p) for p in label_paths]
    lut = build_global_color_lut(label_paths, seed=seed)
    color_frames = [colorize_label(load_label_png(p), lut) for p in label_paths]
    frame_names = [p.stem for p in label_paths]
    strip = _assemble_strip(color_frames, layout=layout)

    fig, ax = _figure_for_strip(strip, len(color_frames), layout=layout, figsize_scale=figsize_scale)
    if show_frame_names:
        _draw_frame_names(ax, frame_names, color_frames[0].shape, layout=layout)
    fig.tight_layout(pad=0)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    return strip, lut, fig, ax


def visualize_prediction_strip(
    pred_dir,
    num_frames=None,
    interval=None,
    seed=0,
    figsize_scale=4,
    show_frame_names=True,
    layout="horizontal",
    save_path=None,
    dpi=200,
):
    """Visualize sampled prediction PNGs as one strip."""
    sampled_pngs, sampled_indices = get_sampled_pngs(pred_dir, num_frames=num_frames, interval=interval)
    strip, lut, fig, ax = visualize_label_strip(
        sampled_pngs,
        seed=seed,
        figsize_scale=figsize_scale,
        show_frame_names=show_frame_names,
        layout=layout,
        save_path=save_path,
        dpi=dpi,
    )
    return strip, lut, sampled_pngs, sampled_indices, fig, ax


def visualize_gt_strip_matched_to_prediction(
    gt_dir,
    pred_sampled_indices,
    original_interval,
    seed=0,
    figsize_scale=4,
    show_frame_names=True,
    layout="horizontal",
    save_path=None,
    dpi=200,
):
    """Visualize GT frames matched to sampled prediction frames."""
    sampled_gt_pngs, gt_indices = get_matched_gt_pngs(gt_dir, pred_sampled_indices, original_interval)
    strip, lut, fig, ax = visualize_label_strip(
        sampled_gt_pngs,
        seed=seed,
        figsize_scale=figsize_scale,
        show_frame_names=show_frame_names,
        layout=layout,
        save_path=save_path,
        dpi=dpi,
    )
    return strip, lut, sampled_gt_pngs, gt_indices, fig, ax


def visualize_rgb_strip_matched_to_prediction(
    frame_dir,
    pred_sampled_indices,
    original_interval,
    figsize_scale=4,
    show_frame_names=False,
    layout="horizontal",
    save_path=None,
    dpi=200,
):
    """Visualize RGB frames matched to sampled prediction frames."""
    sampled_frame_paths, frame_indices = get_matched_frame_paths(
        frame_dir,
        pred_sampled_indices,
        original_interval,
    )
    rgb_frames = [load_rgb_image(p) for p in sampled_frame_paths]
    frame_names = [p.stem for p in sampled_frame_paths]
    strip = _assemble_strip(rgb_frames, layout=layout)

    fig, ax = _figure_for_strip(strip, len(rgb_frames), layout=layout, figsize_scale=figsize_scale)
    if show_frame_names:
        _draw_frame_names(ax, frame_names, rgb_frames[0].shape, layout=layout)
    fig.tight_layout(pad=0)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    return strip, sampled_frame_paths, frame_indices, fig, ax


def main():
    parser = argparse.ArgumentParser(description="Create RGB/GT/prediction qualitative strips.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pred_parser = subparsers.add_parser("prediction")
    pred_parser.add_argument("--pred_dir", required=True)
    pred_parser.add_argument("--num_frames", type=int)
    pred_parser.add_argument("--interval", type=int)
    pred_parser.add_argument("--layout", choices=["horizontal", "vertical"], default="horizontal")
    pred_parser.add_argument("--save_path")

    gt_parser = subparsers.add_parser("gt")
    gt_parser.add_argument("--gt_dir", required=True)
    gt_parser.add_argument("--pred_indices", nargs="+", type=int, required=True)
    gt_parser.add_argument("--original_interval", type=int, required=True)
    gt_parser.add_argument("--layout", choices=["horizontal", "vertical"], default="horizontal")
    gt_parser.add_argument("--save_path")

    rgb_parser = subparsers.add_parser("rgb")
    rgb_parser.add_argument("--frame_dir", required=True)
    rgb_parser.add_argument("--pred_indices", nargs="+", type=int, required=True)
    rgb_parser.add_argument("--original_interval", type=int, required=True)
    rgb_parser.add_argument("--layout", choices=["horizontal", "vertical"], default="horizontal")
    rgb_parser.add_argument("--save_path")

    args = parser.parse_args()

    if args.command == "prediction":
        visualize_prediction_strip(
            args.pred_dir,
            num_frames=args.num_frames,
            interval=args.interval,
            layout=args.layout,
            save_path=args.save_path,
        )
    elif args.command == "gt":
        visualize_gt_strip_matched_to_prediction(
            args.gt_dir,
            np.asarray(args.pred_indices),
            args.original_interval,
            layout=args.layout,
            save_path=args.save_path,
        )
    elif args.command == "rgb":
        visualize_rgb_strip_matched_to_prediction(
            args.frame_dir,
            np.asarray(args.pred_indices),
            args.original_interval,
            layout=args.layout,
            save_path=args.save_path,
        )

    plt.show()


if __name__ == "__main__":
    main()
