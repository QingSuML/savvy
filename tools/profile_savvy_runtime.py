#!/usr/bin/env python3
"""
Run Savvy behavior/runtime profiling over ScanNet videos and plot the logs.

Example:
    python tools/profile_savvy_runtime.py \
        --device cuda:0 \
        --output runtime_logs/scannet_val/
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_CSV = EVAL_ROOT / "tools" / "savvy_behavior_all_scenes.csv"
DEFAULT_GT_DIR = Path("/path/to/scannet/gt")
DEFAULT_VIDEO_DIR = Path("/path/to/scannet/scannet_val")
DEFAULT_SAM1_CKPT = Path("/path/to/sam_vit_h_4b8939.pth")
DEFAULT_SAM2_CKPT = Path("/path/to/sam2.1_hiera_large.pt")
DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def parse_device_to_gpu_id(device: str) -> str:
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    if device == "cuda":
        return "0"
    if device.isdigit():
        return device
    raise ValueError(f"Unsupported device {device!r}. Use cuda:0, cuda:1, or a GPU index.")


def read_ranked_scene_table(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {metrics_csv}")
    if "Scene ID" not in rows[0] or "VPQ_inf" not in rows[0]:
        raise RuntimeError(f"{metrics_csv} must contain 'Scene ID' and 'VPQ_inf' columns")

    rows = sorted(rows, key=lambda r: float(r["VPQ_inf"]), reverse=True)
    ranked = []
    for rank, row in enumerate(rows, start=1):
        ranked.append({
            "rank": str(rank),
            "Scene ID": row["Scene ID"],
            "VPQ_inf": row["VPQ_inf"],
        })
    return ranked


def write_scene_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "Scene ID", "VPQ_inf"])
        writer.writeheader()
        writer.writerows(rows)


def write_scene_txt(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(row["Scene ID"] + "\n")


def prepare_scene_splits(output: Path, metrics_csv: Path, filtered_min_vpq: float, top_k: int) -> dict[str, Path]:
    ranked = read_ranked_scene_table(metrics_csv)
    filtered = [row for row in ranked if float(row["VPQ_inf"]) >= filtered_min_vpq]
    low_tail = [row for row in ranked if float(row["VPQ_inf"]) < filtered_min_vpq]
    top = ranked[:top_k]

    metadata_dir = output / "metadata"
    paths = {
        "allscenes_csv": metadata_dir / "savvy_runtime_all_scenes.csv",
        "allscenes_txt": metadata_dir / "savvy_runtime_all_scenes.txt",
        "filtered_csv": metadata_dir / "savvy_runtime_filtered_scenes.csv",
        "filtered_txt": metadata_dir / "savvy_runtime_filtered_scenes.txt",
        "low_tail_csv": metadata_dir / "savvy_runtime_low_vpq_tail_scenes.csv",
        "top_csv": metadata_dir / f"savvy_runtime_top{top_k}_scenes.csv",
    }

    write_scene_csv(paths["allscenes_csv"], ranked)
    write_scene_txt(paths["allscenes_txt"], ranked)
    write_scene_csv(paths["filtered_csv"], filtered)
    write_scene_txt(paths["filtered_txt"], filtered)
    write_scene_csv(paths["low_tail_csv"], low_tail)
    write_scene_csv(paths["top_csv"], top)

    print(f"Scene split: all={len(ranked)}, filtered={len(filtered)}, low_vpq_tail={len(low_tail)}, top{top_k}={len(top)}")
    if low_tail:
        print("Low-VPQ tail:", ", ".join(row["Scene ID"] for row in low_tail))
    return paths


def run_savvy_behavior_runner(args: argparse.Namespace, all_scenes_csv: Path) -> None:
    gpu_id = parse_device_to_gpu_id(args.device)
    with all_scenes_csv.open(newline="") as f:
        scenes = [row["Scene ID"] for row in csv.DictReader(f)]

    cmd = [
        sys.executable,
        str(EVAL_ROOT / "tools" / "savvy_behavior_runner.py"),
        "--gt_dir", str(args.gt_dir),
        "--video_dir", str(args.video_dir),
        "--sam1_ckpt", str(args.sam1_ckpt),
        "--sam2_ckpt", str(args.sam2_ckpt),
        "--sam2_cfg", str(args.sam2_cfg),
        "--eval_dir", str(args.output / "eval_predictions"),
        "--vis_dir", str(args.output / "visualizations"),
        "--behavior_log_dir", str(args.output / "behavior_logs"),
        "--gpu", gpu_id,
        "--sam_points", str(args.sam_points),
        "--segmenter_stride", str(args.segmenter_stride),
        "--k_keep", "0",
        "--buffer_size", str(args.buffer_size),
        "--margin", str(args.margin),
        "--handshake_min_agreement_hits", str(args.handshake_min_agreement_hits),
        "--handshake_min_visible_frames_for_promotion", str(args.handshake_min_visible_frames_for_promotion),
        "--handshake_min_max_area_for_promotion", str(args.handshake_min_max_area_for_promotion),
        "--handshake_min_max_area_ratio_for_promotion", str(args.handshake_min_max_area_ratio_for_promotion),
        "--scene_names", *scenes,
        "--exp_tag", args.exp_tag,
        "--no_vis",
    ]
    if args.force_rerun:
        cmd.append("--force_rerun")
    if args.resume_from:
        cmd.extend(["--resume_from", args.resume_from])

    print("Running Savvy behavior runner...")
    print(" ".join(cmd[:8]) + f" ... --scene_names <{len(scenes)} scenes> --exp_tag {args.exp_tag}")
    subprocess.run(cmd, cwd=EVAL_ROOT, check=True)


def behavior_log_tag(args: argparse.Namespace) -> str:
    tags = []
    if args.buffer_size != 30:
        tags.append(f"buffer_{args.buffer_size}")
    tags.append("Savvy")
    if args.exp_tag:
        tags.append(args.exp_tag)
    return "__".join(tags)


def load_behavior_experiment(spec: dict) -> dict:
    import numpy as np
    import pandas as pd

    targets = pd.read_csv(spec["target_csv"])
    frames = []
    missing = []
    for scene_id in targets["Scene ID"].tolist():
        log_path = spec["log_root"] / scene_id / "savvy_behavior_stats.csv"
        if not log_path.exists():
            missing.append(scene_id)
            continue
        scene_df = pd.read_csv(log_path)
        scene_df["scene_id"] = scene_id
        scene_df["vpq_inf_rank"] = int(targets.loc[targets["Scene ID"] == scene_id, "rank"].iloc[0])
        scene_df["vpq_inf"] = float(targets.loc[targets["Scene ID"] == scene_id, "VPQ_inf"].iloc[0])
        frames.append(scene_df)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["scene_id", "frame_ordinal"]).reset_index(drop=True)
        df["progress"] = df.groupby("scene_id")["frame_ordinal"].transform(
            lambda s: s / max(float(s.max()), 1.0)
        )
        df["cumulative_newly_discovered_mask_count"] = df.groupby("scene_id")["newly_discovered_mask_count"].cumsum()
        df["object_set_delta"] = df.groupby("scene_id")["object_set_size"].transform(lambda s: s - s.iloc[0])
        query_df = df[df["segmenter_invoked"] == 1].copy()
        query_df["segmenter_invocation_ordinal"] = query_df.groupby("scene_id").cumcount()
        query_df["segmenter_invocation_progress"] = query_df.groupby("scene_id")["segmenter_invocation_ordinal"].transform(
            lambda s: s / max(float(s.max()), 1.0)
        )
        query_df["cumulative_newly_discovered_mask_count"] = query_df.groupby("scene_id")["newly_discovered_mask_count"].cumsum()
    else:
        df = pd.DataFrame()
        query_df = pd.DataFrame()

    loaded = df["scene_id"].nunique() if not df.empty else 0
    print(f"{spec['slug']}: loaded {loaded}/{len(targets)} scenes")
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        print(f"{spec['slug']}: missing {len(missing)} logs: {preview}{suffix}")

    return {**spec, "targets": targets, "df": df, "query_df": query_df, "missing": missing}


def robust_overlay_ylim(exp_df, overlay_avg, metric: str, quantile: float) -> dict | None:
    import numpy as np

    trace_values = exp_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
    mean_values = overlay_avg[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if trace_values.empty and mean_values.empty:
        return None

    trace_cap = float(trace_values.quantile(quantile)) if not trace_values.empty else 0.0
    mean_cap = float(mean_values.max()) if not mean_values.empty else 0.0
    raw_max = float(trace_values.max()) if not trace_values.empty else mean_cap
    upper = max(trace_cap, mean_cap) * 1.08
    if upper <= 0:
        upper = raw_max if raw_max > 0 else 1.0
    return {
        "metric": metric,
        "quantile": quantile,
        "trace_quantile_cap": trace_cap,
        "mean_max": mean_cap,
        "raw_max": raw_max,
        "plot_ymax": upper,
        "visually_clipped": bool(raw_max > upper),
    }


def plot_overlay_traces(exp: dict, plot_dir: Path, overlay_ylim_quantile: float) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = exp["df"]
    if df.empty:
        print(f"Skipping {exp['slug']} overlay: no logs loaded")
        return

    bins = np.linspace(0, 1, 101)
    mids = (bins[:-1] + bins[1:]) / 2
    overlay_metrics = [
        "cumulative_tracking_fps",
        "object_set_size",
        "cumulative_newly_discovered_mask_count",
        "total_mask_count",
        "cuda_device_used_mb",
    ]
    overlay_avg = df.copy()
    overlay_avg["progress_bin"] = pd.cut(overlay_avg["progress"], bins=bins, labels=mids, include_lowest=True)
    overlay_avg = overlay_avg.groupby("progress_bin", observed=True)[overlay_metrics].mean().reset_index()
    overlay_avg["progress_bin"] = overlay_avg["progress_bin"].astype(float)

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    line_alpha = 0.25 if exp["slug"] == "top20" else 0.08
    for _, scene_df in df.groupby("scene_id", sort=False):
        scene_df = scene_df.sort_values("frame_ordinal")
        axes[0].plot(scene_df["progress"], scene_df["cumulative_tracking_fps"], lw=1, alpha=line_alpha)
        axes[1].plot(scene_df["progress"], scene_df["object_set_size"], lw=1, alpha=line_alpha)
        axes[2].plot(scene_df["progress"], scene_df["cumulative_newly_discovered_mask_count"], lw=1, alpha=line_alpha)
        axes[3].plot(scene_df["progress"], scene_df["total_mask_count"], lw=1, alpha=line_alpha)
        axes[4].plot(scene_df["progress"], scene_df["cuda_device_used_mb"], lw=1, alpha=line_alpha)

    mean_style = {"color": "#555555", "lw": 3.0, "alpha": 0.95, "label": "mean"}
    for ax, metric in zip(axes, overlay_metrics):
        ax.plot(overlay_avg["progress_bin"], overlay_avg[metric], **mean_style)

    cap_records = []
    for ax, metric in zip(axes, overlay_metrics):
        cap = robust_overlay_ylim(df, overlay_avg, metric, overlay_ylim_quantile)
        if cap is not None:
            cap_records.append(cap)
            ax.set_ylim(bottom=0, top=cap["plot_ymax"])
    if cap_records:
        pd.DataFrame(cap_records).to_csv(plot_dir / f"{exp['slug']}_overlay_y_axis_caps.csv", index=False)

    axes[0].set_ylabel("Cumulative FPS")
    axes[1].set_ylabel("Object set size")
    axes[2].set_ylabel("Cumulative new")
    axes[3].set_ylabel("Total masks/frame")
    axes[4].set_ylabel("GPU memory MB")
    axes[4].set_xlabel("Normalized video progress")
    for ax in axes:
        ax.legend(loc="upper left")
    fig.suptitle(
        f"Savvy behavior traces across {exp['title']}\n"
        f"Overlay y-axis capped at p{int(overlay_ylim_quantile * 100)}; data and mean are unchanged",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(plot_dir / f"{exp['slug']}_overlay_traces.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_average_behavior(exp: dict, plot_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = exp["df"]
    query_df = exp["query_df"]
    if df.empty:
        print(f"Skipping {exp['slug']} averages: no logs loaded")
        return

    bins = np.linspace(0, 1, 101)
    mids = (bins[:-1] + bins[1:]) / 2
    metrics = [
        "cumulative_tracking_fps",
        "object_set_size",
        "object_set_delta",
        "propagated_mask_count",
        "total_mask_count",
        "cuda_device_used_mb",
        "cuda_memory_reserved_mb",
        "cuda_memory_allocated_mb",
    ]
    avg = df.copy()
    avg["progress_bin"] = pd.cut(avg["progress"], bins=bins, labels=mids, include_lowest=True)
    avg = avg.groupby("progress_bin", observed=True)[metrics].mean().reset_index()
    avg["progress_bin"] = avg["progress_bin"].astype(float)

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    axes[0].plot(avg["progress_bin"], avg["cumulative_tracking_fps"], color="#1f77b4", lw=2)
    axes[0].set_ylabel("Mean cumulative FPS")
    axes[1].plot(avg["progress_bin"], avg["object_set_size"], color="#2ca02c", lw=2)
    axes[1].set_ylabel("Mean object set")
    axes[2].plot(avg["progress_bin"], avg["object_set_delta"], label="object set delta", color="#d62728", lw=2)
    axes[2].set_ylabel("Mean cumulative new")
    axes[2].legend(loc="upper left")
    axes[3].plot(avg["progress_bin"], avg["total_mask_count"], label="total", color="#111111", lw=2)
    axes[3].plot(avg["progress_bin"], avg["propagated_mask_count"], label="propagated", color="#9467bd", lw=2)
    axes[3].set_ylabel("Mean masks/frame")
    axes[3].legend(loc="upper left")
    axes[4].plot(avg["progress_bin"], avg["cuda_device_used_mb"], label="device used", color="#8c564b", lw=2)
    axes[4].plot(avg["progress_bin"], avg["cuda_memory_reserved_mb"], label="process reserved", color="#17becf", lw=2)
    axes[4].plot(avg["progress_bin"], avg["cuda_memory_allocated_mb"], label="process allocated", color="#bcbd22", lw=2)
    axes[4].set_ylabel("Mean GPU memory MB")
    axes[4].set_xlabel("Normalized video progress")
    axes[4].legend(loc="upper left")
    fig.suptitle(f"Average Savvy behavior over {exp['title']}", y=1.01)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{exp['slug']}_average_behavior.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    avg.to_csv(plot_dir / f"{exp['slug']}_average_behavior.csv", index=False)

    discovery_summary = df.groupby("scene_id").agg(
        frames=("frame_ordinal", "count"),
        discovered_total=("newly_discovered_mask_count", "sum"),
        object_set_start=("object_set_size", "first"),
        object_set_end=("object_set_size", "last"),
        segmenter_invocations=("segmenter_invoked", "sum"),
    ).reset_index()
    discovery_summary["object_set_delta"] = discovery_summary["object_set_end"] - discovery_summary["object_set_start"]
    if not query_df.empty:
        query_summary = query_df.groupby("scene_id").agg(
            max_new_per_segmenter_call=("newly_discovered_mask_count", "max"),
            nonzero_segmenter_calls=("newly_discovered_mask_count", lambda s: int((s > 0).sum())),
            mean_new_per_segmenter_call=("newly_discovered_mask_count", "mean"),
        ).reset_index()
        discovery_summary = discovery_summary.merge(query_summary, on="scene_id", how="left")
    discovery_summary.to_csv(plot_dir / f"{exp['slug']}_discovery_summary.csv", index=False)


def plot_discovery_by_invocation(exp: dict, plot_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    query_df = exp["query_df"]
    if query_df.empty:
        print(f"Skipping {exp['slug']} discovery-by-invocation: no segmenter rows loaded")
        return

    bins = np.linspace(0, 1, 101)
    mids = (bins[:-1] + bins[1:]) / 2
    query_avg = query_df.copy()
    query_avg["invocation_progress_bin"] = pd.cut(
        query_avg["segmenter_invocation_progress"], bins=bins, labels=mids, include_lowest=True
    )
    query_metrics = [
        "newly_discovered_mask_count",
        "cumulative_newly_discovered_mask_count",
        "segmenter_candidate_mask_count",
    ]
    query_avg = query_avg.groupby("invocation_progress_bin", observed=True)[query_metrics].mean().reset_index()
    query_avg["invocation_progress_bin"] = query_avg["invocation_progress_bin"].astype(float)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(query_avg["invocation_progress_bin"], query_avg["newly_discovered_mask_count"], color="#d62728", lw=2)
    axes[0].set_ylabel("Mean new/call")
    axes[1].plot(query_avg["invocation_progress_bin"], query_avg["cumulative_newly_discovered_mask_count"], color="#2ca02c", lw=2)
    axes[1].set_ylabel("Mean cumulative new")
    axes[2].plot(query_avg["invocation_progress_bin"], query_avg["segmenter_candidate_mask_count"], color="#9467bd", lw=2)
    axes[2].set_ylabel("Mean candidates/call")
    axes[2].set_xlabel("Normalized segmenter invocation progress")
    fig.suptitle(f"Savvy discovery behavior over segmenter invocations: {exp['title']}", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{exp['slug']}_discovery_by_invocation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    query_avg.to_csv(plot_dir / f"{exp['slug']}_discovery_by_invocation.csv", index=False)


def summarize_sequence(scene_df):
    import numpy as np
    import pandas as pd

    scene_df = scene_df.sort_values("frame_ordinal")
    tracking_seconds = scene_df["frame_tracking_seconds"].sum()
    segmenter_rows = scene_df[scene_df["segmenter_invoked"] == 1]
    return pd.Series({
        "frames": len(scene_df),
        "tracking_seconds": tracking_seconds,
        "sequence_avg_fps": len(scene_df) / tracking_seconds if tracking_seconds > 0 else np.nan,
        "mean_instant_fps": scene_df["instant_tracking_fps"].mean(),
        "mean_object_set_size": scene_df["object_set_size"].mean(),
        "final_object_set_size": scene_df["object_set_size"].iloc[-1],
        "object_set_delta": scene_df["object_set_size"].iloc[-1] - scene_df["object_set_size"].iloc[0],
        "mean_active_object_count": scene_df["active_object_count"].mean(),
        "mean_established_object_count": scene_df["established_object_count"].mean(),
        "mean_transient_object_count": scene_df["transient_object_count"].mean(),
        "mean_propagated_masks_per_frame": scene_df["propagated_mask_count"].mean(),
        "mean_total_masks_per_frame": scene_df["total_mask_count"].mean(),
        "total_new_objects": scene_df["newly_discovered_mask_count"].sum(),
        "segmenter_invocations": scene_df["segmenter_invoked"].sum(),
        "mean_new_objects_per_segmenter_call": segmenter_rows["newly_discovered_mask_count"].mean() if len(segmenter_rows) else np.nan,
        "mean_segmenter_candidates_per_call": segmenter_rows["segmenter_candidate_mask_count"].mean() if len(segmenter_rows) else np.nan,
        "mean_cuda_device_used_mb": scene_df["cuda_device_used_mb"].mean(),
        "mean_cuda_memory_reserved_mb": scene_df["cuda_memory_reserved_mb"].mean(),
        "mean_cuda_memory_allocated_mb": scene_df["cuda_memory_allocated_mb"].mean(),
        "peak_cuda_device_used_mb": scene_df["cuda_device_used_mb"].max(),
        "peak_cuda_memory_reserved_mb": scene_df["cuda_memory_reserved_mb"].max(),
        "peak_cuda_memory_allocated_mb": scene_df["cuda_memory_allocated_mb"].max(),
    })


def write_sequence_summaries(exp: dict, plot_dir: Path) -> None:
    df = exp["df"]
    if df.empty:
        print(f"Skipping {exp['slug']} sequence summary: no logs loaded")
        return
    sequence_summary = df.groupby("scene_id", sort=False).apply(summarize_sequence).reset_index()
    sequence_summary = sequence_summary.merge(
        exp["targets"].rename(columns={"Scene ID": "scene_id"}), on="scene_id", how="left"
    )
    ordered_cols = ["rank", "scene_id", "VPQ_inf"] + [
        c for c in sequence_summary.columns if c not in {"rank", "scene_id", "VPQ_inf"}
    ]
    sequence_summary = sequence_summary[ordered_cols]
    sequence_summary.to_csv(plot_dir / f"{exp['slug']}_sequence_level_summary.csv", index=False)

    numeric_cols = sequence_summary.select_dtypes(include="number").columns.drop("rank")
    overall_summary = sequence_summary[numeric_cols].agg(["mean", "std", "min", "max"]).reset_index()
    overall_summary = overall_summary.rename(columns={"index": "stat"})
    overall_summary.to_csv(plot_dir / f"{exp['slug']}_overall_sequence_summary.csv", index=False)


def plot_runtime_outputs(args: argparse.Namespace, split_paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    log_root = args.output / "behavior_logs" / behavior_log_tag(args)
    plot_dir = args.output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "slug": "allscenes",
            "title": args.all_scenes_title,
            "target_csv": split_paths["allscenes_csv"],
            "log_root": log_root,
        },
        {
            "slug": "filtered",
            "title": f"filtered scenes (VPQ_inf >= {args.filtered_min_vpq:g})",
            "target_csv": split_paths["filtered_csv"],
            "log_root": log_root,
        },
        {
            "slug": f"top{args.top_k}",
            "title": f"top-{args.top_k} VPQ_inf scenes",
            "target_csv": split_paths["top_csv"],
            "log_root": log_root,
        },
    ]

    experiments = [load_behavior_experiment(spec) for spec in specs]
    for exp in experiments:
        plot_overlay_traces(exp, plot_dir, args.overlay_ylim_quantile)
        plot_average_behavior(exp, plot_dir)
        plot_discovery_by_invocation(exp, plot_dir)
        write_sequence_summaries(exp, plot_dir)
    print(f"Saved plots and summaries to {plot_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Savvy runtime behavior over ScanNet videos.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0")
    parser.add_argument("--output", type=Path, required=True, help="Output root for logs, metadata, plots, and eval predictions.")

    parser.add_argument("--metrics_csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--gt_dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--video_dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--sam1_ckpt", type=Path, default=DEFAULT_SAM1_CKPT)
    parser.add_argument("--sam2_ckpt", type=Path, default=DEFAULT_SAM2_CKPT)
    parser.add_argument("--sam2_cfg", default=DEFAULT_SAM2_CFG)

    parser.add_argument("--exp_tag", default="scannet_val_runtime")
    parser.add_argument("--filtered_min_vpq", type=float, default=0.1)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--all_scenes_title", default="all official ScanNet scenes")
    parser.add_argument("--overlay_ylim_quantile", type=float, default=0.95)

    parser.add_argument("--sam_points", type=int, default=32)
    parser.add_argument("--segmenter_stride", type=int, default=3)
    parser.add_argument("--buffer_size", type=int, default=30)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--handshake_min_agreement_hits", type=int, default=5)
    parser.add_argument("--handshake_min_visible_frames_for_promotion", type=int, default=3)
    parser.add_argument("--handshake_min_max_area_for_promotion", type=int, default=-1)
    parser.add_argument("--handshake_min_max_area_ratio_for_promotion", type=float, default=-1.0)

    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--plots_only", action="store_true", help="Skip inference and regenerate plots from existing behavior logs.")
    parser.add_argument("--skip_plots", action="store_true", help="Run inference only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    split_paths = prepare_scene_splits(
        output=args.output,
        metrics_csv=args.metrics_csv,
        filtered_min_vpq=args.filtered_min_vpq,
        top_k=args.top_k,
    )

    if not args.plots_only:
        run_savvy_behavior_runner(args, split_paths["allscenes_csv"])
    if not args.skip_plots:
        plot_runtime_outputs(args, split_paths)


if __name__ == "__main__":
    main()
