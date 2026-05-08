import os
import json
import csv
import time
import copy
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

from evaluation.oga_metrics import (
    compute_sequence_stq_vpq,
    evaluate_vos_consistency,
)


IGNORE_LABEL = -1
ENTITY_CLASS_ID = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="VIPSeg evaluation aligned with finalized Savvy metrics"
    )
    parser.add_argument(
        "--submit_dir", "-i", type=str, required=True,
        help="Prediction directory containing pred.json and pan_pred/<video_id>/<frame>.png"
    )
    parser.add_argument(
        "--truth_dir", type=str,
        default="/path/to/VIPSeg_720P/panomasksRGB",
        help="GT panoptic RGB directory"
    )
    parser.add_argument(
        "--pan_gt_json_file", type=str,
        default="/path/to/VIPSeg_720P/panoptic_gt_VIPSeg_val.json",
        help="GT JSON file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save results; defaults to submit_dir"
    )
    parser.add_argument("--max_pattern_size", type=int, default=3)
    parser.add_argument("--cooccur_sim_thr", type=float, default=0.5)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--ios_thr", type=float, default=0.5)
    return parser.parse_args()


def rgb_to_id(rgb):
    rgb = np.asarray(rgb, dtype=np.uint32)
    return rgb[:, :, 0] + rgb[:, :, 1] * 256 + rgb[:, :, 2] * 256 * 256


def convert_to_class_agnostic(json_data):
    json_data = copy.deepcopy(json_data)
    json_data["categories"] = [{
        "id": ENTITY_CLASS_ID,
        "name": "entity",
        "isthing": 1,
    }]

    for video_ann in json_data["annotations"]:
        for frame_ann in video_ann["annotations"]:
            for segment in frame_ann["segments_info"]:
                segment["category_id"] = ENTITY_CLASS_ID

    return json_data


def build_video_annotation_lookup(json_data):
    out = {}
    for ann in json_data["annotations"]:
        out[ann["video_id"]] = ann["annotations"]
    return out


def build_compact_id_map(frame_annotations):
    ids = []
    seen = set()
    for frame_ann in frame_annotations:
        for seg in frame_ann["segments_info"]:
            sid = seg["id"]
            if sid not in seen:
                ids.append(sid)
                seen.add(sid)
    return {sid: idx for idx, sid in enumerate(ids)}


def collect_video_instance_maps(
    video_id,
    video_meta,
    gt_frame_anns,
    pred_frame_anns,
    truth_dir,
    submit_dir,
    ignore_label=IGNORE_LABEL,
):
    """
    Build per-frame plain instance maps.

    Convention:
      - ignore/unlabeled = 255
      - valid instance IDs = 0, 1, 2, ...
    """
    gt_images = video_meta["images"]

    gt_id_map = build_compact_id_map(gt_frame_anns)
    pred_id_map = build_compact_id_map(pred_frame_anns)

    gt_list = []
    pred_list = []

    for img_meta, gt_ann, pred_ann in zip(gt_images, gt_frame_anns, pred_frame_anns):
        fname = img_meta["file_name"]

        gt_path = os.path.join(truth_dir, video_id, fname)
        pred_path = os.path.join(submit_dir, "pan_pred", video_id, fname)

        gt_pan = rgb_to_id(Image.open(gt_path))
        pred_pan = rgb_to_id(Image.open(pred_path))

        if gt_pan.shape != pred_pan.shape:
            raise ValueError(
                f"Shape mismatch for {video_id}/{fname}: "
                f"GT {gt_pan.shape} vs Pred {pred_pan.shape}"
            )

        gt_inst = np.full(gt_pan.shape, fill_value=ignore_label, dtype=np.int32)
        pred_inst = np.full(pred_pan.shape, fill_value=ignore_label, dtype=np.int32)

        for seg in gt_ann["segments_info"]:
            sid = seg["id"]
            gt_inst[gt_pan == sid] = gt_id_map[sid]

        for seg in pred_ann["segments_info"]:
            sid = seg["id"]
            pred_inst[pred_pan == sid] = pred_id_map[sid]

        gt_list.append(gt_inst)
        pred_list.append(pred_inst)

    return pred_list, gt_list


# Local VPQ decomposition helpers.
def mean_consecutive_jaccard(active_sets):
    if len(active_sets) <= 1:
        return 1.0
    vals = []
    for s1, s2 in zip(active_sets[:-1], active_sets[1:]):
        union = len(s1 | s2)
        if union == 0:
            continue
        vals.append(len(s1 & s2) / union)
    return float(np.mean(vals)) if vals else 1.0


def build_support_chain(gid, assigned_preds, frame_range, inter_lookup, gt_area_lookup):
    chain = []
    for f in frame_range:
        if gid not in gt_area_lookup[f]:
            continue
        active = {p for p in assigned_preds if inter_lookup[f].get((gid, p), 0) > 0}
        if not active:
            continue
        support_mass = sum(inter_lookup[f].get((gid, p), 0) for p in active)
        chain.append((f, active, support_mass))
    return chain


def sever_ratios_from_chain(gid, chain, inter_lookup, sever_ratio_threshold=0.5):
    ratios = []
    flags = []
    if len(chain) <= 1:
        return ratios, flags

    for (f1, s1, _), (f2, s2, _) in zip(chain[:-1], chain[1:]):
        persistent = s1 & s2
        union_ids = s1 | s2

        numer = 0
        denom = 0
        for p in union_ids:
            w = inter_lookup[f1].get((gid, p), 0) + inter_lookup[f2].get((gid, p), 0)
            denom += w
            if p in persistent:
                numer += w

        ratio = (numer / denom) if denom > 0 else 1.0
        ratios.append(ratio)
        flags.append(ratio <= sever_ratio_threshold)

    return ratios, flags


def dominant_fragment_from_chain(gid, chain, inter_lookup, sever_ratio_threshold=0.5):
    if len(chain) == 0:
        return [], set()

    sever_ratios, sever_flags = sever_ratios_from_chain(
        gid, chain, inter_lookup, sever_ratio_threshold=sever_ratio_threshold
    )

    fragments = []
    start = 0
    for idx, is_sever in enumerate(sever_flags):
        if is_sever:
            fragments.append((start, idx))
            start = idx + 1
    fragments.append((start, len(chain) - 1))

    fragment_info = []
    for a, b in fragments:
        frames = [chain[k][0] for k in range(a, b + 1)]
        pred_ids = set()
        mass = 0
        for k in range(a, b + 1):
            f, active, _ = chain[k]
            pred_ids |= active
            mass += sum(inter_lookup[f].get((gid, p), 0) for p in active)
        fragment_info.append({
            "frames": frames,
            "pred_ids": pred_ids,
            "mass": mass,
            "length": len(frames),
        })

    fragment_info.sort(key=lambda x: (x["mass"], x["length"]), reverse=True)
    dom = fragment_info[0]
    return dom["frames"], dom["pred_ids"]


def score_gt_with_fragment(
    gid,
    g_area,
    selected_preds,
    selected_frames,
    pred_area_lookup,
    inter_lookup,
    area_lookup_for_gt,
    area_penalty_power=1.0,
):
    if not selected_preds or not selected_frames:
        return {
            "final_iou": 0.0,
            "matched_preds_for_gt": set(),
        }

    virtual_tpa = 0
    for f in selected_frames:
        virtual_tpa += sum(inter_lookup[f].get((gid, p), 0) for p in selected_preds)

    virtual_p_area = 0
    for f in selected_frames:
        virtual_p_area += sum(pred_area_lookup[f].get(p, 0) for p in selected_preds if p in pred_area_lookup[f])

    union = g_area + virtual_p_area - virtual_tpa
    raw_iou = virtual_tpa / max(union, 1)

    active_sets = []
    for f in selected_frames:
        if gid not in area_lookup_for_gt[f]:
            continue
        active = {p for p in selected_preds if inter_lookup[f].get((gid, p), 0) > 0}
        if active:
            active_sets.append(active)

    soft_stability = mean_consecutive_jaccard(active_sets)
    area_penalty = min(1.0, g_area / max(virtual_p_area, 1))
    area_penalty = area_penalty ** area_penalty_power
    consistency_penalty = soft_stability * area_penalty
    final_iou = raw_iou * consistency_penalty

    matched_preds_for_gt = set()
    for f in selected_frames:
        for p in selected_preds:
            if inter_lookup[f].get((gid, p), 0) > 0:
                matched_preds_for_gt.add(p)

    return {
        "final_iou": final_iou,
        "matched_preds_for_gt": matched_preds_for_gt,
    }


def compute_vpq_decomp_only(
    preds,
    gts,
    ignore_label=IGNORE_LABEL,
    vpq_iou_threshold=0.5,
    k_values=(0, 5, 15, 25, 35, float("inf")),
    ios_threshold=0.5,
    area_penalty_power=1.0,
    sever_ratio_threshold=0.5,
):
    frame_gt_areas = []
    frame_pred_areas = []
    frame_intersections = []

    for p, g in zip(preds, gts):
        p = np.asarray(p)
        g = np.asarray(g)

        p_flat, g_flat = p.ravel(), g.ravel()

        g_valid_mask = (g_flat != ignore_label)
        p_valid_mask = (p_flat != ignore_label) & g_valid_mask

        p_ids, p_counts = np.unique(p_flat[p_valid_mask], return_counts=True)
        cur_p_areas = dict(zip(p_ids.tolist(), p_counts.tolist()))
        frame_pred_areas.append(cur_p_areas)

        g_ids, g_counts = np.unique(g_flat[g_valid_mask], return_counts=True)
        cur_g_areas = dict(zip(g_ids.tolist(), g_counts.tolist()))
        frame_gt_areas.append(cur_g_areas)

        cur_inter = defaultdict(int)
        if p_valid_mask.any():
            for gid, pid in zip(g_flat[p_valid_mask], p_flat[p_valid_mask]):
                cur_inter[(int(gid), int(pid))] += 1
        frame_intersections.append(cur_inter)

    num_frames = len(preds)
    out = {}

    for k in k_values:
        actual_k = 1 if k == 0 else (num_frames if k == float("inf") else min(k, num_frames))

        window_pqs = []
        window_rqs = []
        window_sqs = []

        for t in range(num_frames - actual_k + 1):
            win_gt_areas = defaultdict(int)
            win_pred_areas = defaultdict(int)
            win_inter = defaultdict(int)

            for f in range(t, t + actual_k):
                for gid, a in frame_gt_areas[f].items():
                    win_gt_areas[gid] += a
                for pid, a in frame_pred_areas[f].items():
                    win_pred_areas[pid] += a
                for pair, a in frame_intersections[f].items():
                    win_inter[pair] += a

            win_pred_to_gt = {}
            for pid, p_area in win_pred_areas.items():
                best_gt, max_ios = None, 0.0
                for gid in win_gt_areas.keys():
                    inter = win_inter.get((gid, pid), 0)
                    if inter > 0:
                        ios = inter / max(p_area, 1)
                        if ios > max_ios and ios >= ios_threshold:
                            max_ios, best_gt = ios, gid
                if best_gt is not None:
                    win_pred_to_gt[pid] = best_gt

            win_gt_virtual_preds = defaultdict(list)
            for pid, gid in win_pred_to_gt.items():
                win_gt_virtual_preds[gid].append(pid)

            tp, iou_sum = 0, 0.0
            matched_gts, matched_preds = set(), set()
            window_range = range(t, t + actual_k)

            for gid, g_area in win_gt_areas.items():
                assigned_preds = win_gt_virtual_preds.get(gid, [])
                if not assigned_preds:
                    continue

                chain = build_support_chain(
                    gid, assigned_preds, window_range, frame_intersections, frame_gt_areas
                )
                dom_frames, dom_preds = dominant_fragment_from_chain(
                    gid, chain, frame_intersections, sever_ratio_threshold=sever_ratio_threshold
                )

                gt_score = score_gt_with_fragment(
                    gid,
                    g_area,
                    dom_preds,
                    dom_frames,
                    frame_pred_areas,
                    frame_intersections,
                    frame_gt_areas,
                    area_penalty_power=area_penalty_power,
                )

                iou = gt_score["final_iou"]
                if iou > vpq_iou_threshold:
                    tp += 1
                    iou_sum += iou
                    matched_gts.add(gid)
                    matched_preds |= gt_score["matched_preds_for_gt"]

            fp = 0
            for pid in win_pred_areas.keys():
                if pid not in matched_preds:
                    inter_with_valid_gts = sum(win_inter.get((gid, pid), 0) for gid in win_gt_areas.keys())
                    void_area = win_pred_areas[pid] - inter_with_valid_gts
                    if void_area / max(win_pred_areas[pid], 1) <= 0.5:
                        fp += 1

            fn = len(win_gt_areas) - len(matched_gts)

            denom = tp + 0.5 * fp + 0.5 * fn
            pq = iou_sum / denom if denom > 0 else 0.0
            rq = tp / denom if denom > 0 else 0.0
            sq = iou_sum / tp if tp > 0 else 0.0

            window_pqs.append(pq)
            window_rqs.append(rq)
            window_sqs.append(sq)

        key = "VPQ_inf" if k == float("inf") else f"VPQ_{k}"
        out[key] = float(np.mean(window_pqs)) if window_pqs else 0.0
        out[f"{key}_RQ"] = float(np.mean(window_rqs)) if window_rqs else 0.0
        out[f"{key}_SQ"] = float(np.mean(window_sqs)) if window_sqs else 0.0

    return out


def aggregate_macro(per_video_results):
    if not per_video_results:
        return {}

    def mean_safe(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else 0.0

    video_ids = sorted(per_video_results.keys())

    macro = {
        "num_videos": len(video_ids),
        "baselines": {},
        "identity_persistence": {},
        "temporal_stability": {},
    }

    baseline_keys = [
        "STQ", "AQ", "GQ", "VPQ_avg",
        "VPQ_0", "VPQ_0_RQ", "VPQ_0_SQ",
        "VPQ_5", "VPQ_5_RQ", "VPQ_5_SQ",
        "VPQ_15", "VPQ_15_RQ", "VPQ_15_SQ",
        "VPQ_25", "VPQ_25_RQ", "VPQ_25_SQ",
        "VPQ_35", "VPQ_35_RQ", "VPQ_35_SQ",
        "VPQ_inf", "VPQ_inf_RQ", "VPQ_inf_SQ",
    ]
    for k in baseline_keys:
        macro["baselines"][k] = mean_safe([
            per_video_results[vid].get("baselines", {}).get(k, None)
            for vid in video_ids
        ])

    ip_keys = [
        "combined", "prediction_axis", "gt_axis",
        "ic_p", "ic_g", "pred_spatial", "temporal_bleed", "discovery_tax"
    ]
    for k in ip_keys:
        macro["identity_persistence"][k] = mean_safe([
            per_video_results[vid].get("identity_persistence", {}).get(k, None)
            for vid in video_ids
        ])

    ts_keys = ["combined", "cluster"]
    for k in ts_keys:
        macro["temporal_stability"][k] = mean_safe([
            per_video_results[vid].get("temporal_stability", {}).get(k, None)
            for vid in video_ids
        ])

    patt_keys = set()
    for vid in video_ids:
        patt_keys.update(
            per_video_results[vid].get("temporal_stability", {}).get("pattern_curve", {}).keys()
        )

    patt_curve = {}
    for k in sorted(patt_keys, key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x):
        patt_curve[k] = mean_safe([
            per_video_results[vid].get("temporal_stability", {}).get("pattern_curve", {}).get(k, None)
            for vid in video_ids
        ])
    macro["temporal_stability"]["pattern_curve"] = patt_curve

    return macro


def save_results(output_dir, per_video_results, macro_results, args):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    json_path = os.path.join(output_dir, f"vipseg_savvy_oga_full_results_{timestamp}.json")
    csv_path = os.path.join(output_dir, f"vipseg_savvy_oga_macro_summary_{timestamp}.csv")
    txt_path = os.path.join(output_dir, f"vipseg_savvy_oga_readout_{timestamp}.txt")

    payload = {
        "settings": {
            "submit_dir": args.submit_dir,
            "truth_dir": args.truth_dir,
            "pan_gt_json_file": args.pan_gt_json_file,
            "ignore_label": IGNORE_LABEL,
            "iou_thr": args.iou_thr,
            "ios_thr": args.ios_thr,
            "cooccur_sim_thr": args.cooccur_sim_thr,
            "max_pattern_size": args.max_pattern_size,
        },
        "macro": macro_results,
        "per_video": per_video_results,
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Video ID", "STQ", "AQ", "GQ",
            "VPQ_0", "VPQ_0_RQ", "VPQ_0_SQ",
            "VPQ_5", "VPQ_5_RQ", "VPQ_5_SQ",
            "VPQ_15", "VPQ_15_RQ", "VPQ_15_SQ",
            "VPQ_25", "VPQ_25_RQ", "VPQ_25_SQ",
            "VPQ_35", "VPQ_35_RQ", "VPQ_35_SQ",
            "VPQ_inf", "VPQ_inf_RQ", "VPQ_inf_SQ",
            "IP(C)", "IP(P)", "IP(G)", "IC(P)", "IC(G)", "Spat.IP", "T.Bleed",
            "TS(C)", "Cluster", "Patt1", "Patt2", "Patt3"
        ])

        for vid, res in sorted(per_video_results.items()):
            base = res.get("baselines", {})
            ip = res.get("identity_persistence", {})
            ts = res.get("temporal_stability", {})
            curve = ts.get("pattern_curve", {})

            patt1 = curve.get(1, curve.get("1", 0.0))
            patt2 = curve.get(2, curve.get("2", 0.0))
            patt3 = curve.get(3, curve.get("3", 0.0))

            writer.writerow([
                vid,
                f"{base.get('STQ', 0.0):.4f}",
                f"{base.get('AQ', 0.0):.4f}",
                f"{base.get('GQ', 0.0):.4f}",
                f"{base.get('VPQ_0', 0.0):.4f}",
                f"{base.get('VPQ_0_RQ', 0.0):.4f}",
                f"{base.get('VPQ_0_SQ', 0.0):.4f}",
                f"{base.get('VPQ_5', 0.0):.4f}",
                f"{base.get('VPQ_5_RQ', 0.0):.4f}",
                f"{base.get('VPQ_5_SQ', 0.0):.4f}",
                f"{base.get('VPQ_15', 0.0):.4f}",
                f"{base.get('VPQ_15_RQ', 0.0):.4f}",
                f"{base.get('VPQ_15_SQ', 0.0):.4f}",
                f"{base.get('VPQ_25', 0.0):.4f}",
                f"{base.get('VPQ_25_RQ', 0.0):.4f}",
                f"{base.get('VPQ_25_SQ', 0.0):.4f}",
                f"{base.get('VPQ_35', 0.0):.4f}",
                f"{base.get('VPQ_35_RQ', 0.0):.4f}",
                f"{base.get('VPQ_35_SQ', 0.0):.4f}",
                f"{base.get('VPQ_inf', 0.0):.4f}",
                f"{base.get('VPQ_inf_RQ', 0.0):.4f}",
                f"{base.get('VPQ_inf_SQ', 0.0):.4f}",
                f"{ip.get('combined', 0.0):.4f}",
                f"{ip.get('prediction_axis', 0.0):.4f}",
                f"{ip.get('gt_axis', 0.0):.4f}",
                f"{ip.get('ic_p', 0.0):.4f}",
                f"{ip.get('ic_g', 0.0):.4f}",
                f"{ip.get('pred_spatial', 0.0):.4f}",
                f"{ip.get('temporal_bleed', 0.0):.4f}",
                f"{ts.get('combined', 0.0):.4f}",
                f"{ts.get('cluster', 0.0):.4f}",
                f"{patt1:.4f}",
                f"{patt2:.4f}",
                f"{patt3:.4f}",
            ])

    with open(txt_path, "w") as f:
        base = macro_results.get("baselines", {})
        ip = macro_results.get("identity_persistence", {})
        ts = macro_results.get("temporal_stability", {})
        curve = ts.get("pattern_curve", {})

        f.write("=" * 80 + "\n")
        f.write("VIPSeg evaluation with finalized Savvy metrics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"submit_dir: {args.submit_dir}\n")
        f.write(f"truth_dir: {args.truth_dir}\n")
        f.write(f"pan_gt_json_file: {args.pan_gt_json_file}\n")
        f.write(f"ignore_label: {IGNORE_LABEL}\n\n")

        f.write("Macro results\n")
        f.write("-" * 80 + "\n")
        f.write(f"STQ        : {base.get('STQ', 0.0):.4f}\n")
        f.write(f"AQ         : {base.get('AQ', 0.0):.4f}\n")
        f.write(f"GQ         : {base.get('GQ', 0.0):.4f}\n")
        f.write(f"VPQ_0      : {base.get('VPQ_0', 0.0):.4f}\n")
        f.write(f"VPQ_0_RQ   : {base.get('VPQ_0_RQ', 0.0):.4f}\n")
        f.write(f"VPQ_0_SQ   : {base.get('VPQ_0_SQ', 0.0):.4f}\n")
        f.write(f"VPQ_5      : {base.get('VPQ_5', 0.0):.4f}\n")
        f.write(f"VPQ_5_RQ   : {base.get('VPQ_5_RQ', 0.0):.4f}\n")
        f.write(f"VPQ_5_SQ   : {base.get('VPQ_5_SQ', 0.0):.4f}\n")
        f.write(f"VPQ_15     : {base.get('VPQ_15', 0.0):.4f}\n")
        f.write(f"VPQ_15_RQ  : {base.get('VPQ_15_RQ', 0.0):.4f}\n")
        f.write(f"VPQ_15_SQ  : {base.get('VPQ_15_SQ', 0.0):.4f}\n")
        f.write(f"VPQ_25     : {base.get('VPQ_25', 0.0):.4f}\n")
        f.write(f"VPQ_25_RQ  : {base.get('VPQ_25_RQ', 0.0):.4f}\n")
        f.write(f"VPQ_25_SQ  : {base.get('VPQ_25_SQ', 0.0):.4f}\n")
        f.write(f"VPQ_35     : {base.get('VPQ_35', 0.0):.4f}\n")
        f.write(f"VPQ_35_RQ  : {base.get('VPQ_35_RQ', 0.0):.4f}\n")
        f.write(f"VPQ_35_SQ  : {base.get('VPQ_35_SQ', 0.0):.4f}\n")
        f.write(f"VPQ_inf    : {base.get('VPQ_inf', 0.0):.4f}\n")
        f.write(f"VPQ_inf_RQ : {base.get('VPQ_inf_RQ', 0.0):.4f}\n")
        f.write(f"VPQ_inf_SQ : {base.get('VPQ_inf_SQ', 0.0):.4f}\n\n")

        f.write(f"IP(C)      : {ip.get('combined', 0.0):.4f}\n")
        f.write(f"IP(P)      : {ip.get('prediction_axis', 0.0):.4f}\n")
        f.write(f"IP(G)      : {ip.get('gt_axis', 0.0):.4f}\n")
        f.write(f"IC(P)      : {ip.get('ic_p', 0.0):.4f}\n")
        f.write(f"IC(G)      : {ip.get('ic_g', 0.0):.4f}\n")
        f.write(f"Spat.IP    : {ip.get('pred_spatial', 0.0):.4f}\n")
        f.write(f"T.Bleed    : {ip.get('temporal_bleed', 0.0):.4f}\n\n")

        f.write(f"TS(C)      : {ts.get('combined', 0.0):.4f}\n")
        f.write(f"Cluster    : {ts.get('cluster', 0.0):.4f}\n")
        for k in sorted(curve.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x):
            f.write(f"Patt{k}      : {curve[k]:.4f}\n")

    return json_path, csv_path, txt_path


def print_per_video_summary(per_video_results):
    if not per_video_results:
        print("No per-video results.")
        return

    h = "-" * 255
    print(
        h + "\n" +
        f"{'Video ID':<18} | "
        f"{'STQ':<7} | "
        f"{'AQ':<7} | "
        f"{'GQ':<7} | "
        f"{'VPQ_0':<7} | "
        f"{'0_RQ':<7} | "
        f"{'0_SQ':<7} | "
        f"{'VPQ_5':<7} | "
        f"{'VPQ_15':<7} | "
        f"{'VPQ_25':<7} | "
        f"{'VPQ_35':<7} | "
        f"{'VPQ_inf':<7} | "
        f"{'inf_RQ':<7} | "
        f"{'inf_SQ':<7} | "
        f"{'IP(C)':<7} | "
        f"{'IP(P)':<7} | "
        f"{'IP(G)':<7} | "
        f"{'IC(P)':<7} | "
        f"{'IC(G)':<7} | "
        f"{'Spat.IP':<7} | "
        f"{'T.Bleed':<7} | "
        f"{'TS(C)':<7} | "
        f"{'Cluster':<8} | "
        f"{'Patt1':<8} | "
        f"{'Patt2':<8} | "
        f"{'Patt3':<8}\n" +
        h
    )

    for vid, res in sorted(per_video_results.items()):
        base = res.get("baselines", {})
        ip = res.get("identity_persistence", {})
        ts = res.get("temporal_stability", {})
        curve = ts.get("pattern_curve", {})

        patt1 = curve.get(1, curve.get("1", 0.0))
        patt2 = curve.get(2, curve.get("2", 0.0))
        patt3 = curve.get(3, curve.get("3", 0.0))

        print(
            f"{vid:<18} | "
            f"{base.get('STQ', 0.0):<7.4f} | "
            f"{base.get('AQ', 0.0):<7.4f} | "
            f"{base.get('GQ', 0.0):<7.4f} | "
            f"{base.get('VPQ_0', 0.0):<7.4f} | "
            f"{base.get('VPQ_0_RQ', 0.0):<7.4f} | "
            f"{base.get('VPQ_0_SQ', 0.0):<7.4f} | "
            f"{base.get('VPQ_5', 0.0):<7.4f} | "
            f"{base.get('VPQ_15', 0.0):<7.4f} | "
            f"{base.get('VPQ_25', 0.0):<7.4f} | "
            f"{base.get('VPQ_35', 0.0):<7.4f} | "
            f"{base.get('VPQ_inf', 0.0):<7.4f} | "
            f"{base.get('VPQ_inf_RQ', 0.0):<7.4f} | "
            f"{base.get('VPQ_inf_SQ', 0.0):<7.4f} | "
            f"{ip.get('combined', 0.0):<7.4f} | "
            f"{ip.get('prediction_axis', 0.0):<7.4f} | "
            f"{ip.get('gt_axis', 0.0):<7.4f} | "
            f"{ip.get('ic_p', 0.0):<7.4f} | "
            f"{ip.get('ic_g', 0.0):<7.4f} | "
            f"{ip.get('pred_spatial', 0.0):<7.4f} | "
            f"{ip.get('temporal_bleed', 0.0):<7.4f} | "
            f"{ts.get('combined', 0.0):<7.4f} | "
            f"{ts.get('cluster', 0.0):<8.4f} | "
            f"{patt1:<8.4f} | "
            f"{patt2:<8.4f} | "
            f"{patt3:<8.4f}"
        )
    print(h)


def print_macro_summary(macro_results):
    if not macro_results:
        print("No macro results.")
        return

    base = macro_results.get("baselines", {})
    ip = macro_results.get("identity_persistence", {})
    ts = macro_results.get("temporal_stability", {})
    curve = ts.get("pattern_curve", {})

    print("\n" + "=" * 100)
    print(f"VIPSeg macro summary over {macro_results.get('num_videos', 0)} videos")
    print("=" * 100)
    print(f"STQ        : {base.get('STQ', 0.0):.4f}")
    print(f"AQ         : {base.get('AQ', 0.0):.4f}")
    print(f"GQ         : {base.get('GQ', 0.0):.4f}")
    print(f"VPQ_0      : {base.get('VPQ_0', 0.0):.4f}")
    print(f"VPQ_0_RQ   : {base.get('VPQ_0_RQ', 0.0):.4f}")
    print(f"VPQ_0_SQ   : {base.get('VPQ_0_SQ', 0.0):.4f}")
    print(f"VPQ_5      : {base.get('VPQ_5', 0.0):.4f}")
    print(f"VPQ_5_RQ   : {base.get('VPQ_5_RQ', 0.0):.4f}")
    print(f"VPQ_5_SQ   : {base.get('VPQ_5_SQ', 0.0):.4f}")
    print(f"VPQ_15     : {base.get('VPQ_15', 0.0):.4f}")
    print(f"VPQ_15_RQ  : {base.get('VPQ_15_RQ', 0.0):.4f}")
    print(f"VPQ_15_SQ  : {base.get('VPQ_15_SQ', 0.0):.4f}")
    print(f"VPQ_25     : {base.get('VPQ_25', 0.0):.4f}")
    print(f"VPQ_25_RQ  : {base.get('VPQ_25_RQ', 0.0):.4f}")
    print(f"VPQ_25_SQ  : {base.get('VPQ_25_SQ', 0.0):.4f}")
    print(f"VPQ_35     : {base.get('VPQ_35', 0.0):.4f}")
    print(f"VPQ_35_RQ  : {base.get('VPQ_35_RQ', 0.0):.4f}")
    print(f"VPQ_35_SQ  : {base.get('VPQ_35_SQ', 0.0):.4f}")
    print(f"VPQ_inf    : {base.get('VPQ_inf', 0.0):.4f}")
    print(f"VPQ_inf_RQ : {base.get('VPQ_inf_RQ', 0.0):.4f}")
    print(f"VPQ_inf_SQ : {base.get('VPQ_inf_SQ', 0.0):.4f}")
    print("-" * 100)
    print(f"IP(C)      : {ip.get('combined', 0.0):.4f}")
    print(f"IP(P)      : {ip.get('prediction_axis', 0.0):.4f}")
    print(f"IP(G)      : {ip.get('gt_axis', 0.0):.4f}")
    print(f"IC(P)      : {ip.get('ic_p', 0.0):.4f}")
    print(f"IC(G)      : {ip.get('ic_g', 0.0):.4f}")
    print(f"Spat.IP    : {ip.get('pred_spatial', 0.0):.4f}")
    print(f"T.Bleed    : {ip.get('temporal_bleed', 0.0):.4f}")
    print("-" * 100)
    print(f"TS(C)      : {ts.get('combined', 0.0):.4f}")
    print(f"Cluster    : {ts.get('cluster', 0.0):.4f}")
    for k in sorted(curve.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x):
        print(f"Patt{k:<4}: {curve[k]:.4f}")
    print("=" * 100)


def main():
    args = parse_args()
    submit_dir = args.submit_dir
    truth_dir = args.truth_dir
    gt_json_path = args.pan_gt_json_file
    output_dir = args.output_dir or submit_dir

    if not os.path.isdir(submit_dir):
        raise FileNotFoundError(f"submit_dir does not exist: {submit_dir}")
    if not os.path.isdir(truth_dir):
        raise FileNotFoundError(f"truth_dir does not exist: {truth_dir}")
    if not os.path.isfile(gt_json_path):
        raise FileNotFoundError(f"GT json does not exist: {gt_json_path}")

    pred_json_path = os.path.join(submit_dir, "pred.json")
    if not os.path.isfile(pred_json_path):
        raise FileNotFoundError(f"Prediction json does not exist: {pred_json_path}")

    with open(pred_json_path, "r") as f:
        pred_jsons = json.load(f)
    with open(gt_json_path, "r") as f:
        gt_jsons = json.load(f)

    pred_jsons = convert_to_class_agnostic(pred_jsons)
    gt_jsons = convert_to_class_agnostic(gt_jsons)

    pred_by_video = build_video_annotation_lookup(pred_jsons)
    gt_by_video = build_video_annotation_lookup(gt_jsons)

    per_video_results = {}

    pbar = tqdm(gt_jsons["videos"])
    for video_meta in pbar:
        video_id = video_meta["video_id"]
        pbar.set_description(video_id)

        if video_id not in pred_by_video:
            print(f"[WARN] Missing prediction annotations for video {video_id}, skipping.")
            continue

        gt_frame_anns = gt_by_video[video_id]
        pred_frame_anns = pred_by_video[video_id]

        if len(gt_frame_anns) != len(pred_frame_anns):
            raise ValueError(
                f"Frame annotation count mismatch for {video_id}: "
                f"GT {len(gt_frame_anns)} vs Pred {len(pred_frame_anns)}"
            )

        pred_list, gt_list = collect_video_instance_maps(
            video_id=video_id,
            video_meta=video_meta,
            gt_frame_anns=gt_frame_anns,
            pred_frame_anns=pred_frame_anns,
            truth_dir=truth_dir,
            submit_dir=submit_dir,
            ignore_label=IGNORE_LABEL,
        )

        baselines = compute_sequence_stq_vpq(
            preds=pred_list,
            gts=gt_list,
            ignore_label=IGNORE_LABEL,
            vpq_iou_threshold=0.5,
            k_values=(0, 5, 15, 25, 35, float("inf")),
            ios_threshold=args.ios_thr,
            area_penalty_power=1.0,
            sever_ratio_threshold=0.5,
            use_soft_stability=True,
            use_dominant_fragment=True,
        )

        vpq_decomp = compute_vpq_decomp_only(
            preds=pred_list,
            gts=gt_list,
            ignore_label=IGNORE_LABEL,
            vpq_iou_threshold=0.5,
            k_values=(0, 5, 15, 25, 35, float("inf")),
            ios_threshold=args.ios_thr,
            area_penalty_power=1.0,
            sever_ratio_threshold=0.5,
        )

        baselines.update(vpq_decomp)

        full = evaluate_vos_consistency(
            seg_list=pred_list,
            gt_list=gt_list,
            ignore_label=IGNORE_LABEL,
            iou_thr=args.iou_thr,
            ios_thr=args.ios_thr,
            cooccur_sim_thr=args.cooccur_sim_thr,
            max_pattern_size=args.max_pattern_size,
        )

        full["baselines"] = baselines
        per_video_results[video_id] = full

    macro_results = aggregate_macro(per_video_results)
    json_path, csv_path, txt_path = save_results(
        output_dir, per_video_results, macro_results, args
    )

    print_per_video_summary(per_video_results)
    print_macro_summary(macro_results)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")
    print(f"Saved TXT : {txt_path}")


if __name__ == "__main__":
    main()
