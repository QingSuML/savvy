#!/usr/bin/env python3
"""Support-matrix and OGA behavior-matrix visualizations.

This module contains the reusable parts of ``prediction_support_matrix.ipynb``
and the behavior-topology plot from ``result_summary.ipynb``.  The functions are
usable from notebooks, scripts, or a small CLI.
"""

import argparse
import glob
import os
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, LinearSegmentedColormap
from PIL import Image


DEFAULT_SUPPORT_COLORS = ["#351D70", "#8A6CDF", "#4FECBF"]


def make_support_cmap(name="support_default", n=256):
    """Create one of the custom support-matrix colormaps."""
    name = str(name).lower()
    palettes = {
        "support_default": DEFAULT_SUPPORT_COLORS,
        "teal_purple_indigo_reversed": DEFAULT_SUPPORT_COLORS,
        "teal_purple_indigo": list(reversed(DEFAULT_SUPPORT_COLORS)),
        "teal_indigo": ["#351D70", "#4FECBF"],
        "teal_indigo_white": ["#FFFFFF", "#4FECBF", "#351D70"],
        "white_teal_indigo": ["#FFFFFF", "#4FECBF", "#351D70"],
    }
    if name not in palettes:
        raise ValueError(f"Unknown support colormap: {name}")
    return LinearSegmentedColormap.from_list(name, palettes[name], N=n)


def resolve_support_cmap(cmap="support_default"):
    """Resolve a custom support colormap, matplotlib colormap name, or object."""
    if cmap is None or isinstance(cmap, Colormap):
        return cmap
    if not isinstance(cmap, str):
        raise TypeError(f"cmap must be a string, Colormap, or None. Got {type(cmap)}")
    custom_names = {
        "support_default",
        "teal_purple_indigo_reversed",
        "teal_purple_indigo",
        "teal_indigo",
        "teal_indigo_white",
        "white_teal_indigo",
    }
    if cmap.lower() in custom_names:
        return make_support_cmap(cmap)
    try:
        return mpl.colormaps[cmap]
    except KeyError as exc:
        raise ValueError(f"Unknown matplotlib colormap: {cmap}") from exc


def _to_numpy(x):
    return x if isinstance(x, np.ndarray) else np.asarray(x)


def _numeric_stem(path):
    return int(os.path.splitext(os.path.basename(path))[0])


def _get_frame_masks(arr, ignore_label=0):
    arr = _to_numpy(arr).astype(np.int64)
    ids = np.unique(arr)
    ids = ids[ids != ignore_label]

    masks = {}
    areas = {}
    for sid in ids:
        sid = int(sid)
        mask = arr == sid
        area = int(mask.sum())
        if area > 0:
            masks[sid] = mask
            areas[sid] = area
    return masks, areas


def _collect_global_ids(seg_list, ignore_label=0):
    ids = set()
    for seg in seg_list:
        arr = _to_numpy(seg).astype(np.int64)
        cur = np.unique(arr)
        cur = cur[cur != ignore_label]
        ids.update(int(x) for x in cur)
    ids = sorted(ids)
    return ids, {sid: i for i, sid in enumerate(ids)}


def _find_prediction_scene_folder(pred_root, scene_id):
    direct = os.path.join(pred_root, scene_id)
    if os.path.isdir(direct):
        return direct

    candidates = [
        os.path.join(pred_root, d)
        for d in os.listdir(pred_root)
        if os.path.isdir(os.path.join(pred_root, d)) and d.endswith(scene_id)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return sorted(candidates, key=len)[0]
    raise FileNotFoundError(f"Could not find predictions for scene {scene_id} under {pred_root}")


def _scannet_sampling_rule(total_frames):
    if total_frames <= 300:
        return 1, total_frames
    if total_frames <= 1000:
        return 2, total_frames
    if total_frames <= 1500:
        return 3, total_frames
    return 3, 1500


def load_scannet_scene_for_support_matrix(scene_id, gt_dir, pred_root, resize_pred_to_gt=True):
    """Load ScanNet GT instance PNGs and prediction PNGs using evaluator sampling."""
    gt_scene_dir = os.path.join(gt_dir, scene_id, "instance")
    pred_scene_dir = _find_prediction_scene_folder(pred_root, scene_id)

    gt_files = sorted(glob.glob(os.path.join(gt_scene_dir, "*.png")), key=_numeric_stem)
    pred_map = {_numeric_stem(p): p for p in glob.glob(os.path.join(pred_scene_dir, "*.png"))}
    if not gt_files:
        raise FileNotFoundError(f"No GT PNGs found in {gt_scene_dir}")
    if not pred_map:
        raise FileNotFoundError(f"No prediction PNGs found in {pred_scene_dir}")

    interval, cutoff = _scannet_sampling_rule(len(gt_files))
    gt_files = gt_files[:cutoff]

    preds, gts, frame_indices = [], [], []
    for gt_path in gt_files:
        idx = _numeric_stem(gt_path)
        if idx % interval != 0 or idx not in pred_map:
            continue

        gt_img = Image.open(gt_path)
        pred_img = Image.open(pred_map[idx])
        if resize_pred_to_gt and pred_img.size != gt_img.size:
            resample_filter = getattr(Image, "Resampling", Image).NEAREST
            pred_img = pred_img.resize(gt_img.size, resample=resample_filter)

        gts.append(np.asarray(gt_img).astype(np.int64))
        preds.append(np.asarray(pred_img).astype(np.int64))
        frame_indices.append(idx)

    return preds, gts, frame_indices


def load_hm3d_scene_for_support_matrix(scene_id, gt_dir, pred_root, resize_pred_to_gt=True):
    """Load HM3D semantic ``.npy`` GT and prediction PNGs."""
    gt_scene_dir = os.path.join(gt_dir, scene_id, "semantic")
    pred_scene_dir = _find_prediction_scene_folder(pred_root, scene_id)

    gt_files = sorted(glob.glob(os.path.join(gt_scene_dir, "*.npy")), key=_numeric_stem)
    pred_files = sorted(glob.glob(os.path.join(pred_scene_dir, "*.png")), key=_numeric_stem)
    if not gt_files:
        raise FileNotFoundError(f"No GT npy files found in {gt_scene_dir}")
    if not pred_files:
        raise FileNotFoundError(f"No prediction PNGs found in {pred_scene_dir}")

    n = min(len(gt_files), len(pred_files))
    preds, gts, frame_indices = [], [], []
    for i in range(n):
        gt = np.load(gt_files[i]).astype(np.int64)
        pred_img = Image.open(pred_files[i])
        if resize_pred_to_gt and pred_img.size != (gt.shape[1], gt.shape[0]):
            resample_filter = getattr(Image, "Resampling", Image).NEAREST
            pred_img = pred_img.resize((gt.shape[1], gt.shape[0]), resample=resample_filter)
        gts.append(gt)
        preds.append(np.asarray(pred_img).astype(np.int64))
        frame_indices.append(_numeric_stem(pred_files[i]))

    return preds, gts, frame_indices


def matrix_key_from_normalization(normalize_by="pred", binary=False):
    """Resolve a support dictionary matrix key from a normalization mode."""
    if binary:
        return "E"
    key = str(normalize_by).lower()
    if key in {"pred", "prediction", "a_p"}:
        return "W_pred"
    if key in {"gt", "reference", "ref", "b_g"}:
        return "W_gt"
    if key in {"none", "raw", "intersection", "m"}:
        return "M"
    raise ValueError("normalize_by must be one of: pred, gt, none")


def compute_prediction_reference_support_matrix(
    preds,
    gts,
    ignore_label=0,
    iou_thr=0.5,
    ios_thr=0.5,
    use_void_tolerant=True,
    eps=1e-12,
):
    """Compute prediction-reference support matrices over a video.

    Definitions:
        ``M[p,g]`` is accumulated intersection.
        ``W_pred[p,g] = M[p,g] / A_p[p]`` measures GT support inside a prediction.
        ``W_gt[p,g] = M[p,g] / B_g[g]`` measures GT coverage by a prediction.
        ``E`` is the binary edge matrix using ``IoU >= iou_thr`` or
        ``IoP >= ios_thr`` on accumulated counts.
    """
    if len(preds) != len(gts):
        raise ValueError("preds and gts must have the same length")

    p_ids, p_to_i = _collect_global_ids(preds, ignore_label=ignore_label)
    g_ids, g_to_i = _collect_global_ids(gts, ignore_label=ignore_label)

    M = np.zeros((len(p_ids), len(g_ids)), dtype=np.float64)
    A_p = np.zeros(len(p_ids), dtype=np.float64)
    B_g = np.zeros(len(g_ids), dtype=np.float64)

    for pred_arr, gt_arr in zip(preds, gts):
        pred_arr = _to_numpy(pred_arr).astype(np.int64)
        gt_arr = _to_numpy(gt_arr).astype(np.int64)
        valid_gt = gt_arr != ignore_label

        pred_masks, pred_areas = _get_frame_masks(pred_arr, ignore_label)
        gt_masks, gt_areas = _get_frame_masks(gt_arr, ignore_label)

        for gid, area in gt_areas.items():
            B_g[g_to_i[gid]] += area

        for pid, pmask in pred_masks.items():
            pmask_valid = np.logical_and(pmask, valid_gt) if use_void_tolerant else pmask
            p_area = int(pmask_valid.sum())
            if p_area == 0:
                continue
            p_idx = p_to_i[pid]
            A_p[p_idx] += p_area

            for gid, gmask in gt_masks.items():
                inter = int(np.logical_and(pmask_valid, gmask).sum())
                if inter:
                    M[p_idx, g_to_i[gid]] += inter

    W_pred = M / np.maximum(A_p[:, None], eps)
    W_gt = M / np.maximum(B_g[None, :], eps)
    union = A_p[:, None] + B_g[None, :] - M
    iou = M / np.maximum(union, eps)
    iop = W_pred
    E = np.logical_or(iou >= iou_thr, np.logical_and(iou < iou_thr, iop >= ios_thr)).astype(np.float64)

    row_degree = E.sum(axis=1)
    col_degree = E.sum(axis=0)
    row_ip = W_pred.max(axis=1) if W_pred.size else np.array([])
    col_recall = W_gt.max(axis=0) if W_gt.size else np.array([])
    ic_p = np.divide(1.0, np.maximum(row_degree, 1.0), where=row_degree >= 0)
    ic_g = np.divide(1.0, np.maximum(col_degree, 1.0), where=col_degree >= 0)

    return {
        "M": M,
        "W": W_pred,
        "W_pred": W_pred,
        "W_gt": W_gt,
        "E": E,
        "A_p": A_p,
        "B_g": B_g,
        "p_ids": p_ids,
        "g_ids": g_ids,
        "row_degree": row_degree,
        "col_degree": col_degree,
        "row_ip": row_ip,
        "col_recall": col_recall,
        "ic_p": ic_p,
        "ic_g": ic_g,
        "iou": iou,
    }


def get_support_matrix_order(
    support,
    matrix_key="W_gt",
    sort_mode="dominant_gt",
    min_row_mass=0.0,
    min_col_mass=0.0,
    top_rows=None,
    top_cols=None,
):
    """Compute row/column order for support matrix visualization."""
    mat = support[matrix_key]
    row_mass = mat.sum(axis=1)
    col_mass = mat.sum(axis=0)

    rows = np.where(row_mass >= min_row_mass)[0]
    cols = np.where(col_mass >= min_col_mass)[0]
    if rows.size == 0 or cols.size == 0:
        return rows, cols

    sub = mat[np.ix_(rows, cols)]
    mode = str(sort_mode).lower()

    if mode == "dominant_gt":
        dom_col = np.argmax(sub, axis=1)
        rows = rows[np.lexsort((-sub.max(axis=1), dom_col))]
        sub_after_rows = mat[np.ix_(rows, cols)]
        col_mass_after = sub_after_rows.sum(axis=0)
        cols = cols[np.argsort(-col_mass_after)]
    elif mode == "dominant_pred":
        dom_row = np.argmax(sub, axis=0)
        cols = cols[np.lexsort((-sub.max(axis=0), dom_row))]
        sub_after_cols = mat[np.ix_(rows, cols)]
        row_mass_after = sub_after_cols.sum(axis=1)
        rows = rows[np.argsort(-row_mass_after)]
    elif mode == "mass":
        rows = rows[np.argsort(-row_mass[rows])]
        cols = cols[np.argsort(-col_mass[cols])]
    elif mode in {"id", "none"}:
        pass
    else:
        raise ValueError("sort_mode must be one of: dominant_gt, dominant_pred, mass, id")

    if top_rows is not None:
        rows = rows[:top_rows]
    if top_cols is not None:
        cols = cols[:top_cols]
    return rows, cols


def slice_support(support, row_order, col_order):
    """Slice support dictionary arrays by prediction rows and reference columns."""
    out = {}
    for key in ["M", "W", "W_pred", "W_gt", "E", "iou"]:
        if key in support:
            out[key] = support[key][np.ix_(row_order, col_order)]

    out["p_ids"] = [support["p_ids"][i] for i in row_order]
    out["g_ids"] = [support["g_ids"][j] for j in col_order]
    for key in ["row_degree", "row_ip", "ic_p", "A_p"]:
        if key in support:
            out[key] = support[key][row_order]
    for key in ["col_degree", "col_recall", "ic_g", "B_g"]:
        if key in support:
            out[key] = support[key][col_order]
    return out


def resolve_orientation(orientation="pred_ref", transpose=None):
    """Resolve support matrix display orientation."""
    if transpose is not None:
        return "ref_pred" if transpose else "pred_ref"
    key = str(orientation).lower()
    if key in {"pred_ref", "prediction_reference", "p_g", "pg"}:
        return "pred_ref"
    if key in {"ref_pred", "reference_prediction", "g_p", "gp"}:
        return "ref_pred"
    raise ValueError("orientation must be 'pred_ref' or 'ref_pred'")


def orient_sliced_matrix(sliced, matrix_key, orientation="pred_ref", transpose=None):
    """Return matrix and axis labels in requested display orientation."""
    orientation = resolve_orientation(orientation, transpose=transpose)
    mat = sliced[matrix_key]
    if orientation == "pred_ref":
        return {
            "matrix": mat,
            "x_ids": sliced["g_ids"],
            "y_ids": sliced["p_ids"],
            "xlabel": "Reference instance ID",
            "ylabel": "Prediction identity ID",
        }
    return {
        "matrix": mat.T,
        "x_ids": sliced["p_ids"],
        "y_ids": sliced["g_ids"],
        "xlabel": "Prediction identity ID",
        "ylabel": "Reference instance ID",
    }


def plot_support_matrix(
    support,
    normalize_by="gt",
    binary=False,
    orientation="ref_pred",
    transpose=None,
    sort_mode="dominant_gt",
    min_row_mass=0.0,
    min_col_mass=0.0,
    top_rows=80,
    top_cols=80,
    title=None,
    figsize=(8, 8),
    show_ids=True,
    max_tick_labels=80,
    annotate=False,
    annotation_thr=0.01,
    cmap="support_default",
    save_path=None,
    dpi=200,
):
    """Plot one prediction-reference support matrix."""
    matrix_key = matrix_key_from_normalization(normalize_by, binary=binary)
    row_order, col_order = get_support_matrix_order(
        support,
        matrix_key=matrix_key,
        sort_mode=sort_mode,
        min_row_mass=min_row_mass,
        min_col_mass=min_col_mass,
        top_rows=top_rows,
        top_cols=top_cols,
    )
    sliced = slice_support(support, row_order, col_order)
    oriented = orient_sliced_matrix(sliced, matrix_key, orientation=orientation, transpose=transpose)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(oriented["matrix"], aspect="auto", interpolation="nearest", cmap=resolve_support_cmap(cmap))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(matrix_key)

    if title is None:
        title = f"Prediction-Reference Support Matrix ({matrix_key})"
    ax.set_title(title)
    ax.set_xlabel(oriented["xlabel"])
    ax.set_ylabel(oriented["ylabel"])

    if show_ids and len(oriented["x_ids"]) <= max_tick_labels:
        ax.set_xticks(np.arange(len(oriented["x_ids"])))
        ax.set_xticklabels(oriented["x_ids"], rotation=90, fontsize=8)
    else:
        ax.set_xticks([])
    if show_ids and len(oriented["y_ids"]) <= max_tick_labels:
        ax.set_yticks(np.arange(len(oriented["y_ids"])))
        ax.set_yticklabels(oriented["y_ids"], fontsize=8)
    else:
        ax.set_yticks([])

    if annotate:
        mat = oriented["matrix"]
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                if mat[y, x] >= annotation_thr:
                    ax.text(x, y, f"{mat[y, x]:.2f}", ha="center", va="center", fontsize=6, color="white")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, ax, sliced


def plot_support_degrees(
    support,
    sort_mode="mass",
    min_row_mass=0.0,
    min_col_mass=0.0,
    top_rows=80,
    top_cols=80,
    title="Support Graph Degrees",
    figsize=(10, 4),
    save_path=None,
    dpi=200,
):
    """Plot row and column degrees induced by the support edge matrix."""
    row_order, col_order = get_support_matrix_order(
        support,
        matrix_key="E",
        sort_mode=sort_mode,
        min_row_mass=min_row_mass,
        min_col_mass=min_col_mass,
        top_rows=top_rows,
        top_cols=top_cols,
    )
    sliced = slice_support(support, row_order, col_order)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].bar(np.arange(len(sliced["p_ids"])), sliced["row_degree"], color="#6577D9")
    axes[0].set_title("Prediction degree")
    axes[0].set_xlabel("Prediction IDs")
    axes[0].set_ylabel("# matched references")

    axes[1].bar(np.arange(len(sliced["g_ids"])), sliced["col_degree"], color="#6AD9CE")
    axes[1].set_title("Reference degree")
    axes[1].set_xlabel("Reference IDs")
    axes[1].set_ylabel("# matched predictions")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, axes, sliced


def summarize_support_matrix(support):
    """Return compact scalar diagnostics for a support matrix."""
    row_degree = support["row_degree"]
    col_degree = support["col_degree"]
    return {
        "num_prediction_ids": int(len(support["p_ids"])),
        "num_reference_ids": int(len(support["g_ids"])),
        "num_edges": int(support["E"].sum()),
        "mean_prediction_degree": float(row_degree.mean()) if row_degree.size else 0.0,
        "mean_reference_degree": float(col_degree.mean()) if col_degree.size else 0.0,
        "max_prediction_degree": int(row_degree.max()) if row_degree.size else 0,
        "max_reference_degree": int(col_degree.max()) if col_degree.size else 0,
        "mean_prediction_best_support": float(support["row_ip"].mean()) if support["row_ip"].size else 0.0,
        "mean_reference_best_coverage": float(support["col_recall"].mean()) if support["col_recall"].size else 0.0,
    }


def generate_summary_table(method_csvs):
    """Build a method-by-metric mean table from evaluation CSVs."""
    rows = []
    for method_name, csv_path in method_csvs.items():
        row = pd.read_csv(csv_path).mean(numeric_only=True)
        row.name = method_name
        rows.append(row)
    return pd.DataFrame(rows)


def plot_behavior_matrix(
    method_csvs,
    colors=None,
    ic_p_col="IC (P)",
    ic_g_col="IC (G)",
    threshold=0.2,
    title="OGA Tracking Behavior Topology Matrix",
    save_path=None,
    dpi=200,
):
    """Plot the IC(P)-IC(G) behavior matrix for multiple methods."""
    if colors is None:
        colors = {
            "EntitySAM": "#f2a69d",
            "DEVA+SAM": "#cdf1ab",
            "Savvy": "#aed4ff",
        }

    fig, ax = plt.subplots(figsize=(8, 8))
    for name, csv_path in method_csvs.items():
        df = pd.read_csv(csv_path, usecols=[ic_p_col, ic_g_col])
        color = colors.get(name, "#9E9E9E")
        ax.scatter(
            df[ic_p_col],
            df[ic_g_col],
            label=name,
            color=color,
            alpha=0.72,
            s=56,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.scatter(
            df[ic_p_col].mean(),
            df[ic_g_col].mean(),
            marker="*",
            s=520,
            color=color,
            edgecolors="grey",
            linewidth=1.1,
            zorder=3,
        )

    ax.axvline(x=threshold, color="darkgray", linestyle="--", linewidth=1)
    ax.axhline(y=threshold, color="darkgray", linestyle="--", linewidth=1)

    text_kwargs = dict(fontsize=12, fontweight="bold", color="silver", ha="center")
    ax.text(0.65, 0.60, "Isomorphic", **text_kwargs)
    ax.text(0.65, 0.10, "Shattering", **text_kwargs)
    ax.text(0.10, 0.60, "Merging", **text_kwargs)
    ax.text(0.10, 0.10, "Chaos", **text_kwargs)

    ticks = np.linspace(0.0, 1.0, 6)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Prediction Breadth ($IC_p$) $\rightarrow$ Anti-Merging")
    ax.set_ylabel(r"Ground Truth Breadth ($IC_g$) $\rightarrow$ Anti-Shattering")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(title="Method", frameon=True, loc="upper left")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, ax


def count_scene_masks(pred_scene_dir, ignore_label=0):
    """Count per-frame and unique prediction IDs in a saved scene folder."""
    png_files = sorted(glob.glob(os.path.join(pred_scene_dir, "*.png")), key=_numeric_stem)
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {pred_scene_dir}")

    total_frame_masks = 0
    frame_counts = []
    unique_ids = set()
    for path in png_files:
        arr = np.asarray(Image.open(path)).astype(np.int64)
        ids = np.unique(arr)
        ids = ids[ids != ignore_label]
        frame_counts.append(len(ids))
        total_frame_masks += len(ids)
        unique_ids.update(int(x) for x in ids)

    return {
        "num_frames": len(png_files),
        "total_frame_masks": int(total_frame_masks),
        "unique_prediction_ids": int(len(unique_ids)),
        "frame_counts": frame_counts,
        "unique_ids": sorted(unique_ids),
        "mean_masks_per_frame": float(np.mean(frame_counts)) if frame_counts else 0.0,
        "max_masks_per_frame": int(np.max(frame_counts)) if frame_counts else 0,
        "min_masks_per_frame": int(np.min(frame_counts)) if frame_counts else 0,
    }


def _parse_method_csvs(items):
    method_csvs = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Method CSVs must use NAME=PATH syntax")
        name, path = item.split("=", 1)
        method_csvs[name] = path
    return method_csvs


def main():
    parser = argparse.ArgumentParser(description="Support matrix and behavior matrix visualization.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    support_parser = subparsers.add_parser("support-matrix")
    support_parser.add_argument("--dataset", choices=["scannet", "hm3d"], default="scannet")
    support_parser.add_argument("--scene_id", required=True)
    support_parser.add_argument("--gt_dir", required=True)
    support_parser.add_argument("--pred_root", required=True)
    support_parser.add_argument("--normalize_by", choices=["pred", "gt", "none"], default="gt")
    support_parser.add_argument("--orientation", choices=["pred_ref", "ref_pred"], default="ref_pred")
    support_parser.add_argument("--top_rows", type=int, default=80)
    support_parser.add_argument("--top_cols", type=int, default=80)
    support_parser.add_argument("--save_path")

    behavior_parser = subparsers.add_parser("behavior-matrix")
    behavior_parser.add_argument("--method_csv", nargs="+", required=True, help="NAME=PATH entries")
    behavior_parser.add_argument("--save_path")

    args = parser.parse_args()

    if args.command == "support-matrix":
        if args.dataset == "scannet":
            preds, gts, _ = load_scannet_scene_for_support_matrix(args.scene_id, args.gt_dir, args.pred_root)
        else:
            preds, gts, _ = load_hm3d_scene_for_support_matrix(args.scene_id, args.gt_dir, args.pred_root)
        support = compute_prediction_reference_support_matrix(preds, gts)
        print(summarize_support_matrix(support))
        plot_support_matrix(
            support,
            normalize_by=args.normalize_by,
            orientation=args.orientation,
            top_rows=args.top_rows,
            top_cols=args.top_cols,
            save_path=args.save_path,
        )
    elif args.command == "behavior-matrix":
        plot_behavior_matrix(_parse_method_csvs(args.method_csv), save_path=args.save_path)

    plt.show()


if __name__ == "__main__":
    main()
