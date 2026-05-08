#!/usr/bin/env python3
"""Discovery and reassociation event analysis/visualization for ScanNet.

This module provides ScanNet identity event analysis and
discovery/reassociation curve plotting.  It can be used in two ways:

1. Run event analysis for saved ScanNet predictions.
2. Load saved ``identity_events.json`` files and plot paper-style method curves.
"""

import os
import glob
import csv
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.patheffects as pe


def _to_numpy(seg):
    return np.array(seg) if not isinstance(seg, np.ndarray) else seg


def _get_frame_masks(arr, ignore_label=None):
    arr = arr.astype(np.int64)
    ids = np.unique(arr)
    if ignore_label is not None:
        ids = ids[ids != ignore_label]
    masks, areas = {}, {}
    for sid in ids:
        mask = (arr == sid)
        area = int(mask.sum())
        if area > 0:
            masks[int(sid)], areas[int(sid)] = mask, area
    return masks, areas


def _compute_frame_matches(pred_masks,
                           pred_areas,
                           gt_masks,
                           gt_areas,
                           iou_thr,
                           ios_thr,
                           valid_gt_mask=None):
    frame_matches = {gid: set() for gid in gt_masks.keys()}
    frame_inters = defaultdict(dict)

    for pid, pmask in pred_masks.items():
        pmask_valid = np.logical_and(pmask, valid_gt_mask) if valid_gt_mask is not None else pmask
        p_area = int(pmask_valid.sum())
        if p_area == 0:
            continue

        for gid, gmask in gt_masks.items():
            g_area = gt_areas[gid]
            inter = int(np.logical_and(pmask_valid, gmask).sum())
            if inter == 0:
                continue

            frame_inters[pid][gid] = inter
            union = p_area + g_area - inter
            if union <= 0:
                continue

            iou = inter / union
            iop = inter / p_area

            if iou >= iou_thr or (iou < iou_thr and iop >= ios_thr):
                frame_matches[gid].add(pid)

    return frame_matches, frame_inters


class ScannetVOSDataset:
    def __init__(self, gt_dir, pred_dir):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir

    def get_scene_ids(self):
        if not os.path.exists(self.gt_dir):
            return []
        return sorted([
            d for d in os.listdir(self.gt_dir)
            if os.path.isdir(os.path.join(self.gt_dir, d))
        ])

    def load_scene_frames(self, scene_id):
        gt_path = os.path.join(self.gt_dir, scene_id, "instance")
        pr_path = os.path.join(self.pred_dir, scene_id)

        if not all(os.path.exists(p) for p in [gt_path, pr_path]):
            return [], [], []

        g_files = sorted(
            glob.glob(os.path.join(gt_path, "*.png")),
            key=lambda x: int(os.path.basename(x).split(".")[0])
        )
        p_map = {
            int(os.path.basename(p).split(".")[0]): p
            for p in glob.glob(os.path.join(pr_path, "*.png"))
        }

        total_frames = len(g_files)
        if total_frames <= 300:
            adaptive_interval = 1
            cutoff = total_frames
        elif total_frames <= 1000:
            adaptive_interval = 2
            cutoff = total_frames
        elif total_frames <= 1500:
            adaptive_interval = 3
            cutoff = total_frames
        else:
            adaptive_interval = 3
            cutoff = 1500

        g_files = g_files[:cutoff]
        print(f"[{scene_id}] Frames: {total_frames} -> Capped at {cutoff}, Interval: {adaptive_interval}")

        gl, pl, frame_indices = [], [], []
        for gp in g_files:
            idx = int(os.path.basename(gp).split(".")[0])
            if idx % adaptive_interval == 0 and idx in p_map:
                gi = Image.open(gp)
                pi = Image.open(p_map[idx])

                if pi.size != gi.size:
                    resample_filter = getattr(Image, "Resampling", Image).NEAREST
                    pi = pi.resize(gi.size, resample=resample_filter)

                gl.append(np.array(gi).astype(np.int64))
                pl.append(np.array(pi).astype(np.int64))
                frame_indices.append(idx)

        return pl, gl, frame_indices


def build_global_pred_to_gt(preds, gts, ignore_label=0, ios_threshold=0.5):
    """
    Global unique pred->GT ownership by IoS = inter / pred_area.
    One prediction id can be assigned to at most one GT over the full video.
    """
    global_gt_areas = defaultdict(int)
    global_pred_areas = defaultdict(int)
    global_intersections = defaultdict(lambda: defaultdict(int))

    for p, g in zip(preds, gts):
        p = np.asarray(p)
        g = np.asarray(g)

        p_flat, g_flat = p.ravel(), g.ravel()
        g_valid_mask = (g_flat != ignore_label)
        p_valid_mask = (p_flat != ignore_label) & g_valid_mask

        p_ids, p_counts = np.unique(p_flat[p_valid_mask], return_counts=True)
        for pid, c in zip(p_ids.tolist(), p_counts.tolist()):
            global_pred_areas[int(pid)] += int(c)

        g_ids, g_counts = np.unique(g_flat[g_valid_mask], return_counts=True)
        for gid, c in zip(g_ids.tolist(), g_counts.tolist()):
            global_gt_areas[int(gid)] += int(c)

        if p_valid_mask.any():
            for gid, pid in zip(g_flat[p_valid_mask], p_flat[p_valid_mask]):
                global_intersections[int(gid)][int(pid)] += 1

    global_pred_to_gt = {}
    for pid, p_area in global_pred_areas.items():
        best_gt, max_ios = None, 0.0
        for gid in global_gt_areas.keys():
            inter = global_intersections[gid].get(pid, 0)
            if inter > 0:
                ios = inter / max(p_area, 1)
                if ios > max_ios and ios >= ios_threshold:
                    max_ios, best_gt = ios, gid
        if best_gt is not None:
            global_pred_to_gt[int(pid)] = int(best_gt)

    return global_pred_to_gt


def _windowed_ga_event_for_gid(gid,
                               t0,
                               frame_indices,
                               gt_mask_info_list,
                               pred_mask_info_list,
                               global_pred_to_gt,
                               window_size,
                               support_thr,
                               coverage_thr,
                               strict_ownership=True):
    """
    Generic GA-style event success check for either discovery or reappearance.

    Event anchored at eval frame t0. Within [t0, t0 + window_size], search for a frame
    where the union of all valid support predictions achieves sufficient GT coverage.

    A valid support prediction p for GT g at frame t must satisfy:
        S_t(p, g) = |p_t ∩ g_t| / |p_t| >= support_thr

    If strict_ownership is enabled, it must also satisfy:
        global_pred_to_gt[p] == g

    Event success criterion:
        | g_t ∩ union_{p in P_t(g)} p_t | / |g_t| >= coverage_thr
    """
    T = len(frame_indices)
    window_end = min(T - 1, t0 + window_size)

    union_supporting_pids = set()
    first_success_t = None
    first_success_frame_idx = None
    first_success_pids = []
    first_success_support_scores = {}
    first_success_joint_coverage = None

    best_joint_coverage = 0.0
    best_joint_coverage_t = None
    best_joint_coverage_frame_idx = None
    best_supporting_pids = []
    best_support_scores = {}

    per_frame_window_records = []

    for tw in range(t0, window_end + 1):
        gt_masks, gt_areas = gt_mask_info_list[tw]
        pred_masks, pred_areas = pred_mask_info_list[tw]

        if gid not in gt_masks:
            per_frame_window_records.append({
                "eval_t": int(tw),
                "frame_idx": int(frame_indices[tw]),
                "supporting_pids": [],
                "support_scores": {},
                "joint_coverage": 0.0,
            })
            continue

        gmask = gt_masks[gid]
        g_area = gt_areas[gid]
        if g_area == 0:
            per_frame_window_records.append({
                "eval_t": int(tw),
                "frame_idx": int(frame_indices[tw]),
                "supporting_pids": [],
                "support_scores": {},
                "joint_coverage": 0.0,
            })
            continue

        supporting_pids = []
        support_scores = {}
        union_mask = np.zeros_like(gmask, dtype=bool)

        for pid, pmask in pred_masks.items():
            p_area = pred_areas[pid]
            if p_area == 0:
                continue

            inter = int(np.logical_and(pmask, gmask).sum())
            if inter == 0:
                continue

            support = inter / p_area
            if support >= support_thr:
                if strict_ownership and global_pred_to_gt.get(int(pid), None) != int(gid):
                    continue
                supporting_pids.append(int(pid))
                support_scores[int(pid)] = float(support)
                union_mask |= pmask

        if supporting_pids:
            covered_gt_area = int(np.logical_and(union_mask, gmask).sum())
            joint_coverage = covered_gt_area / g_area
            union_supporting_pids.update(supporting_pids)
        else:
            joint_coverage = 0.0

        per_frame_window_records.append({
            "eval_t": int(tw),
            "frame_idx": int(frame_indices[tw]),
            "supporting_pids": sorted(supporting_pids),
            "support_scores": support_scores,
            "joint_coverage": float(joint_coverage),
        })

        if joint_coverage > best_joint_coverage:
            best_joint_coverage = float(joint_coverage)
            best_joint_coverage_t = int(tw)
            best_joint_coverage_frame_idx = int(frame_indices[tw])
            best_supporting_pids = sorted(supporting_pids)
            best_support_scores = support_scores

        if first_success_t is None and joint_coverage >= coverage_thr:
            first_success_t = int(tw)
            first_success_frame_idx = int(frame_indices[tw])
            first_success_pids = sorted(supporting_pids)
            first_success_support_scores = support_scores
            first_success_joint_coverage = float(joint_coverage)

    success = first_success_t is not None

    return {
        "successful_event": bool(success),
        "window_end_eval_t": int(window_end),
        "window_end_frame_idx": int(frame_indices[window_end]),
        "support_thr": float(support_thr),
        "coverage_thr": float(coverage_thr),
        "strict_ownership": bool(strict_ownership),
        "supporting_pids_within_window": sorted(list(union_supporting_pids)),
        "first_success_eval_t": first_success_t,
        "first_success_frame_idx": first_success_frame_idx,
        "first_success_pids": first_success_pids,
        "first_success_support_scores": first_success_support_scores,
        "first_success_joint_coverage": first_success_joint_coverage,
        "delay_eval": None if first_success_t is None else int(first_success_t - t0),
        "best_joint_coverage": float(best_joint_coverage),
        "best_joint_coverage_eval_t": best_joint_coverage_t,
        "best_joint_coverage_frame_idx": best_joint_coverage_frame_idx,
        "best_supporting_pids": best_supporting_pids,
        "best_support_scores": best_support_scores,
        "window_frame_records": per_frame_window_records,
    }


def analyze_scene_identity_events(preds,
                                  gts,
                                  frame_indices,
                                  ignore_label=0,
                                  iou_thr=0.5,
                                  ios_thr=0.5,
                                  min_gap=1,
                                  discovery_window=0,
                                  reassoc_window=0,
                                  support_thr=0.5,
                                  coverage_thr=0.5,
                                  strict_ownership=True):
    assert len(preds) == len(gts) == len(frame_indices)

    global_pred_to_gt = build_global_pred_to_gt(
        preds, gts, ignore_label=ignore_label, ios_threshold=ios_thr
    )

    T = len(preds)

    prediction_side_frame_ids = []
    gt_present_list = []
    frame_match_list = []
    gt_mask_info_list = []
    pred_mask_info_list = []

    for pred_arr, gt_arr in zip(preds, gts):
        pred_arr = _to_numpy(pred_arr).astype(np.int64)
        gt_arr = _to_numpy(gt_arr).astype(np.int64)

        pred_ids_in_frame = [int(x) for x in np.unique(pred_arr) if int(x) != ignore_label]
        prediction_side_frame_ids.append(pred_ids_in_frame)

        gt_masks, gt_areas = _get_frame_masks(gt_arr, ignore_label)
        gt_mask_info_list.append((gt_masks, gt_areas))
        gt_present_list.append(set(gt_masks.keys()))

        valid_gt_mask = (gt_arr != ignore_label)
        pred_masks_raw, _ = _get_frame_masks(pred_arr, ignore_label)
        pred_masks = {}
        pred_areas = {}
        for pid, pmask in pred_masks_raw.items():
            pmask_valid = np.logical_and(pmask, valid_gt_mask)
            area_valid = int(pmask_valid.sum())
            if area_valid > 0:
                pred_masks[int(pid)] = pmask_valid
                pred_areas[int(pid)] = area_valid
        pred_mask_info_list.append((pred_masks, pred_areas))

        frame_matches, _ = _compute_frame_matches(
            pred_masks, pred_areas, gt_masks, gt_areas,
            iou_thr=iou_thr, ios_thr=ios_thr, valid_gt_mask=None
        )
        frame_match_list.append(frame_matches)

    gt_seen_before = set()
    gt_prev_present = set()
    gt_last_present_eval_t = {}
    scene_events = []

    gt_discovery_total = 0
    gt_discovery_success_total = 0
    gt_discovery_fail_total = 0

    gt_reappear_total = 0
    gt_reassoc_success_total = 0
    gt_reassoc_fail_total = 0

    for t, frame_idx in enumerate(frame_indices):
        gt_present = gt_present_list[t]

        for gid in sorted(gt_present):
            # GT first appearance -> discovery event
            if gid not in gt_seen_before:
                gt_discovery_total += 1

                disc_info = _windowed_ga_event_for_gid(
                    gid=gid,
                    t0=t,
                    frame_indices=frame_indices,
                    gt_mask_info_list=gt_mask_info_list,
                    pred_mask_info_list=pred_mask_info_list,
                    global_pred_to_gt=global_pred_to_gt,
                    window_size=discovery_window,
                    support_thr=support_thr,
                    coverage_thr=coverage_thr,
                    strict_ownership=strict_ownership,
                )

                if disc_info["successful_event"]:
                    gt_discovery_success_total += 1
                else:
                    gt_discovery_fail_total += 1

                scene_events.append({
                    "event_type": "discovery",
                    "eval_t": int(t),
                    "frame_idx": int(frame_idx),
                    "gid": int(gid),
                    "gap_len": 0,
                    "window_end_eval_t": disc_info["window_end_eval_t"],
                    "window_end_frame_idx": disc_info["window_end_frame_idx"],
                    "support_thr": disc_info["support_thr"],
                    "coverage_thr": disc_info["coverage_thr"],
                    "strict_ownership": disc_info["strict_ownership"],
                    "event_window": int(discovery_window),
                    "supporting_pids_within_window": disc_info["supporting_pids_within_window"],
                    "first_success_eval_t": disc_info["first_success_eval_t"],
                    "first_success_frame_idx": disc_info["first_success_frame_idx"],
                    "first_success_pids": disc_info["first_success_pids"],
                    "first_success_support_scores": disc_info["first_success_support_scores"],
                    "first_success_joint_coverage": disc_info["first_success_joint_coverage"],
                    "delay_eval": disc_info["delay_eval"],
                    "best_joint_coverage": disc_info["best_joint_coverage"],
                    "best_joint_coverage_eval_t": disc_info["best_joint_coverage_eval_t"],
                    "best_joint_coverage_frame_idx": disc_info["best_joint_coverage_frame_idx"],
                    "best_supporting_pids": disc_info["best_supporting_pids"],
                    "best_support_scores": disc_info["best_support_scores"],
                    "window_frame_records": disc_info["window_frame_records"],
                    "successful_discovery": disc_info["successful_event"],
                    "successful_reassoc": None,
                    "matched_pids_at_event": disc_info["first_success_pids"],
                    "successful_pids": disc_info["first_success_pids"],
                    "winning_pid": None if len(disc_info["first_success_pids"]) == 0 else int(disc_info["first_success_pids"][0]),
                })

            # GT reappearance after a real gap -> reassociation event
            elif gid not in gt_prev_present:
                prev_seen_t = gt_last_present_eval_t.get(gid, None)
                gap_len = (t - prev_seen_t - 1) if prev_seen_t is not None else 0

                if gap_len >= min_gap:
                    gt_reappear_total += 1

                    reapp_info = _windowed_ga_event_for_gid(
                        gid=gid,
                        t0=t,
                        frame_indices=frame_indices,
                        gt_mask_info_list=gt_mask_info_list,
                        pred_mask_info_list=pred_mask_info_list,
                        global_pred_to_gt=global_pred_to_gt,
                        window_size=reassoc_window,
                        support_thr=support_thr,
                        coverage_thr=coverage_thr,
                        strict_ownership=strict_ownership,
                    )

                    if reapp_info["successful_event"]:
                        gt_reassoc_success_total += 1
                    else:
                        gt_reassoc_fail_total += 1

                    scene_events.append({
                        "event_type": "reappearance",
                        "eval_t": int(t),
                        "frame_idx": int(frame_idx),
                        "gid": int(gid),
                        "gap_len": int(gap_len),
                        "window_end_eval_t": reapp_info["window_end_eval_t"],
                        "window_end_frame_idx": reapp_info["window_end_frame_idx"],
                        "support_thr": reapp_info["support_thr"],
                        "coverage_thr": reapp_info["coverage_thr"],
                        "strict_ownership": reapp_info["strict_ownership"],
                        "event_window": int(reassoc_window),
                        "supporting_pids_within_window": reapp_info["supporting_pids_within_window"],
                        "first_success_eval_t": reapp_info["first_success_eval_t"],
                        "first_success_frame_idx": reapp_info["first_success_frame_idx"],
                        "first_success_pids": reapp_info["first_success_pids"],
                        "first_success_support_scores": reapp_info["first_success_support_scores"],
                        "first_success_joint_coverage": reapp_info["first_success_joint_coverage"],
                        "delay_eval": reapp_info["delay_eval"],
                        "best_joint_coverage": reapp_info["best_joint_coverage"],
                        "best_joint_coverage_eval_t": reapp_info["best_joint_coverage_eval_t"],
                        "best_joint_coverage_frame_idx": reapp_info["best_joint_coverage_frame_idx"],
                        "best_supporting_pids": reapp_info["best_supporting_pids"],
                        "best_support_scores": reapp_info["best_support_scores"],
                        "window_frame_records": reapp_info["window_frame_records"],
                        "successful_discovery": None,
                        "successful_reassoc": reapp_info["successful_event"],
                        "matched_pids_at_event": reapp_info["first_success_pids"],
                        "successful_pids": reapp_info["first_success_pids"],
                        "winning_pid": None if len(reapp_info["first_success_pids"]) == 0 else int(reapp_info["first_success_pids"][0]),
                    })
                else:
                    scene_events.append({
                        "event_type": "short_gap_return",
                        "eval_t": int(t),
                        "frame_idx": int(frame_idx),
                        "gid": int(gid),
                        "gap_len": int(gap_len),
                        "window_end_eval_t": None,
                        "window_end_frame_idx": None,
                        "support_thr": float(support_thr),
                        "coverage_thr": float(coverage_thr),
                        "strict_ownership": bool(strict_ownership),
                        "event_window": int(reassoc_window),
                        "supporting_pids_within_window": [],
                        "first_success_eval_t": None,
                        "first_success_frame_idx": None,
                        "first_success_pids": [],
                        "first_success_support_scores": {},
                        "first_success_joint_coverage": None,
                        "delay_eval": None,
                        "best_joint_coverage": None,
                        "best_joint_coverage_eval_t": None,
                        "best_joint_coverage_frame_idx": None,
                        "best_supporting_pids": [],
                        "best_support_scores": {},
                        "window_frame_records": [],
                        "successful_discovery": None,
                        "successful_reassoc": None,
                        "matched_pids_at_event": [],
                        "successful_pids": [],
                        "winning_pid": None,
                    })

        for gid in gt_present:
            gt_last_present_eval_t[gid] = t

        gt_seen_before.update(gt_present)
        gt_prev_present = gt_present

    scene_summary = {
        "num_eval_frames": int(T),
        "gt_discovery_events": int(gt_discovery_total),
        "successful_discoveries": int(gt_discovery_success_total),
        "failed_discoveries": int(gt_discovery_fail_total),
        "discovery_success_rate": float(gt_discovery_success_total / gt_discovery_total) if gt_discovery_total > 0 else 0.0,
        "gt_reappearance_events": int(gt_reappear_total),
        "successful_reassociations": int(gt_reassoc_success_total),
        "failed_reassociations": int(gt_reassoc_fail_total),
        "reassoc_success_rate": float(gt_reassoc_success_total / gt_reappear_total) if gt_reappear_total > 0 else 0.0,
        "num_globally_assigned_pred_ids": int(len(global_pred_to_gt)),
        "discovery_window": int(discovery_window),
        "reassoc_window": int(reassoc_window),
        "event_support_thr": float(support_thr),
        "event_coverage_thr": float(coverage_thr),
        "strict_ownership": bool(strict_ownership),
    }

    return {
        "summary": scene_summary,
        "events": scene_events,
        "prediction_side_frame_ids": prediction_side_frame_ids,
        "global_pred_to_gt": {int(k): int(v) for k, v in global_pred_to_gt.items()},
    }


def _build_prediction_side_curves(frame_id_list, background_id=0):
    T = len(frame_id_list)
    if T <= 1:
        return None, None, T

    seen_objects, prev_objects = set(), set()
    total_D, total_R = 0, 0
    avg_D_curve, avg_R_curve = [], []

    for frame_ids in frame_id_list:
        current_objects = set([obj for obj in frame_ids if obj != background_id])
        total_D += len(prev_objects - current_objects)
        total_R += len((current_objects.intersection(seen_objects)) - prev_objects)

        seen_objects.update(current_objects)
        num_seen = len(seen_objects)

        avg_D_curve.append(total_D / num_seen if num_seen > 0 else 0.0)
        avg_R_curve.append(total_R / num_seen if num_seen > 0 else 0.0)
        prev_objects = current_objects

    return np.array(avg_D_curve), np.array(avg_R_curve), T


def _build_gt_curves_from_scene_result(scene_result):
    T = int(scene_result["summary"]["num_eval_frames"])
    events = scene_result["events"]

    if T <= 1:
        return None, None, None, None, T

    discovery_by_t = np.zeros(T, dtype=np.float64)
    success_disc_by_t = np.zeros(T, dtype=np.float64)
    reappear_by_t = np.zeros(T, dtype=np.float64)
    success_reassoc_by_t = np.zeros(T, dtype=np.float64)

    cumulative_seen = set()
    denom_curve = np.zeros(T, dtype=np.float64)

    events_by_t = [[] for _ in range(T)]
    for ev in events:
        t = int(ev["eval_t"])
        if 0 <= t < T:
            events_by_t[t].append(ev)

    for t in range(T):
        for ev in events_by_t[t]:
            cumulative_seen.add(int(ev["gid"]))
            if ev["event_type"] == "discovery":
                discovery_by_t[t] += 1.0
                if bool(ev["successful_discovery"]):
                    success_disc_by_t[t] += 1.0
            elif ev["event_type"] == "reappearance":
                reappear_by_t[t] += 1.0
                if bool(ev["successful_reassoc"]):
                    success_reassoc_by_t[t] += 1.0

        denom_curve[t] = max(len(cumulative_seen), 1)

    discovery_curve = np.cumsum(discovery_by_t) / denom_curve
    success_discovery_curve = np.cumsum(success_disc_by_t) / denom_curve
    reappear_curve = np.cumsum(reappear_by_t) / denom_curve
    success_reassoc_curve = np.cumsum(success_reassoc_by_t) / denom_curve

    return discovery_curve, success_discovery_curve, reappear_curve, success_reassoc_curve, T


def _interp_curve(curve, num_grid_points=1000):
    T = len(curve)
    if T <= 1:
        return None
    normalized_grid = np.linspace(0, 1, T)
    target_grid = np.linspace(0, 1, num_grid_points)
    interp = interp1d(normalized_grid, curve, kind="previous", fill_value="extrapolate")
    return interp(target_grid)


def load_identity_events_json(analysis_dir):
    path = os.path.join(analysis_dir, "identity_events.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing identity event file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _load_method_identity_curves(analysis_dir,
                                 event_type,
                                 success_key,
                                 numerator="normalized",
                                 num_grid_points=1000):
    """Load per-scene GT event and success curves for one method."""
    data = load_identity_events_json(analysis_dir)
    all_interp_event = []
    all_interp_success = []

    for scene_result in data.values():
        T = int(scene_result["summary"]["num_eval_frames"])
        events = scene_result["events"]
        if T <= 1:
            continue

        event_by_t = np.zeros(T, dtype=np.float64)
        success_by_t = np.zeros(T, dtype=np.float64)
        denom_curve = np.ones(T, dtype=np.float64)

        events_by_t = [[] for _ in range(T)]
        for ev in events:
            t = int(ev["eval_t"])
            if 0 <= t < T:
                events_by_t[t].append(ev)

        cumulative_seen = set()
        for t in range(T):
            for ev in events_by_t[t]:
                cumulative_seen.add(int(ev["gid"]))
                if ev["event_type"] == event_type:
                    event_by_t[t] += 1.0
                    if bool(ev.get(success_key, False)):
                        success_by_t[t] += 1.0
            if numerator == "normalized":
                denom_curve[t] = max(len(cumulative_seen), 1)

        event_curve = np.cumsum(event_by_t) / denom_curve
        success_curve = np.cumsum(success_by_t) / denom_curve

        interp_event = _interp_curve(event_curve, num_grid_points=num_grid_points)
        interp_success = _interp_curve(success_curve, num_grid_points=num_grid_points)
        if interp_event is not None:
            all_interp_event.append(interp_event)
        if interp_success is not None:
            all_interp_success.append(interp_success)

    if not all_interp_event or not all_interp_success:
        raise ValueError(f"No valid curves found in {analysis_dir}")

    return {
        "event_curves": all_interp_event,
        "success_curves": all_interp_success,
        "mean_event": np.mean(all_interp_event, axis=0),
        "mean_success": np.mean(all_interp_success, axis=0),
    }


def load_method_reassociation_curves(analysis_dir, num_grid_points=1000):
    """Load GT reappearance and successful reassociation curves for one method."""
    return _load_method_identity_curves(
        analysis_dir,
        event_type="reappearance",
        success_key="successful_reassoc",
        numerator="normalized",
        num_grid_points=num_grid_points,
    )


def load_method_discovery_curves(analysis_dir, num_grid_points=1000, normalized=False):
    """Load GT discovery and successful discovery curves for one method."""
    return _load_method_identity_curves(
        analysis_dir,
        event_type="discovery",
        success_key="successful_discovery",
        numerator="normalized" if normalized else "count",
        num_grid_points=num_grid_points,
    )


def plot_multimethod_reassociation_curve(methods,
                                         num_grid_points=1000,
                                         plot_threads=True,
                                         save_path=None,
                                         title="GT Reappearance and Successful Reassociation"):
    """Plot reassociation curves from multiple identity-event analysis folders.

    Args:
        methods: mapping from method name to config. Each config needs
            ``analysis_dir`` and can optionally provide colors.
    """
    grid = np.linspace(0, 1, num_grid_points)
    fig, ax = plt.subplots(figsize=(8, 6))

    default_event_color = "#74E3D8"
    default_event_thread = "#AEEFE8"

    for method_name, cfg in methods.items():
        loaded = load_method_reassociation_curves(cfg["analysis_dir"], num_grid_points=num_grid_points)

        if plot_threads:
            for curve in loaded["event_curves"]:
                ax.plot(grid, curve, color=cfg.get("thread_reappear", default_event_thread),
                        alpha=0.06, linewidth=0.5, zorder=1)
            for curve in loaded["success_curves"]:
                ax.plot(grid, curve, color=cfg.get("thread_success", cfg.get("color_success", "#6577D9")),
                        alpha=0.06, linewidth=0.5, zorder=1)

        ax.plot(grid, loaded["mean_event"],
                label=f"{method_name} GT Reappearances",
                color=cfg.get("color_reappear", default_event_color),
                linewidth=3,
                path_effects=[pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])
        ax.plot(grid, loaded["mean_success"],
                label=f"{method_name} Successful Reassoc.",
                color=cfg.get("color_success", "#6577D9"),
                linewidth=3,
                path_effects=[pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])

    ax.set_title(title, fontsize=14, fontweight="bold", color="#4a4a4a")
    ax.set_xlabel("Normalized Time ($t/T$)", fontsize=12, color="#4a4a4a")
    ax.set_ylabel("Accumulated Events / Number of Seen GT Objects", fontsize=12, color="#4a4a4a")
    ax.set_xlim([0, 1])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_multimethod_discovery_curve(methods,
                                     num_grid_points=1000,
                                     normalized=False,
                                     plot_threads=True,
                                     save_path=None,
                                     title="GT Discovery and Successful Discovery"):
    """Plot discovery curves from multiple identity-event analysis folders."""
    grid = np.linspace(0, 1, num_grid_points)
    fig, ax = plt.subplots(figsize=(8, 6))

    default_event_color = "#74E3D8"
    default_event_thread = "#AEEFE8"

    for method_name, cfg in methods.items():
        loaded = load_method_discovery_curves(
            cfg["analysis_dir"],
            num_grid_points=num_grid_points,
            normalized=normalized,
        )

        if plot_threads:
            for curve in loaded["event_curves"]:
                ax.plot(grid, curve, color=cfg.get("thread_discovery", default_event_thread),
                        alpha=0.06, linewidth=0.5, zorder=1)
            for curve in loaded["success_curves"]:
                ax.plot(grid, curve, color=cfg.get("thread_success", cfg.get("color_success", "#7F74FF")),
                        alpha=0.06, linewidth=0.5, zorder=1)

        ax.plot(grid, loaded["mean_event"],
                label=f"{method_name} GT Discoveries",
                color=cfg.get("color_discovery", default_event_color),
                linewidth=3,
                path_effects=[pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])
        ax.plot(grid, loaded["mean_success"],
                label=f"{method_name} Successful Discoveries",
                color=cfg.get("color_success", "#7F74FF"),
                linewidth=3,
                path_effects=[pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])

    ylabel = "Accumulated Events / Number of Seen GT Objects" if normalized else "Accumulated Events"
    ax.set_title(title, fontsize=14, fontweight="bold", color="#4a4a4a")
    ax.set_xlabel("Normalized Time ($t/T$)", fontsize=12, color="#4a4a4a")
    ax.set_ylabel(ylabel, fontsize=12, color="#4a4a4a")
    ax.set_xlim([0, 1])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_prediction_reappearance_curve(scene_results, background_id=0, num_grid_points=1000, save_path=None,
                                       title="ScanNet: Prediction-side Disappearance & Reappearance"):
    grid = np.linspace(0, 1, num_grid_points)
    plt.figure(figsize=(10, 6))

    all_interp_D = []
    all_interp_R = []

    for result in scene_results.values():
        d_curve, r_curve, _ = _build_prediction_side_curves(result["prediction_side_frame_ids"], background_id=background_id)
        if d_curve is None:
            continue
        all_interp_D.append(_interp_curve(d_curve, num_grid_points))
        all_interp_R.append(_interp_curve(r_curve, num_grid_points))

    if not all_interp_D:
        print("No valid sequences for plotting.")
        return

    mean_D = np.mean(all_interp_D, axis=0)
    mean_R = np.mean(all_interp_R, axis=0)

    for curve in all_interp_D:
        plt.plot(grid, curve, color="#9088FF", alpha=0.03, linewidth=0.5, zorder=1)
    for curve in all_interp_R:
        plt.plot(grid, curve, color="#90CEC6", alpha=0.03, linewidth=0.5, zorder=1)

    plt.plot(grid, mean_D, label="Avg Disappearances", color="#7A6DFF", linewidth=3, zorder=10,
             path_effects=[pe.Stroke(linewidth=6, foreground="white"), pe.Normal()])
    plt.plot(grid, mean_R, label="Avg Reappearances", color="#6AD9CE", linewidth=3, zorder=10,
             path_effects=[pe.Stroke(linewidth=6, foreground="white"), pe.Normal()])

    plt.title(title, fontsize=14, fontweight="bold", color="#4a4a4a")
    plt.xlabel("Normalized Time ($t/T$)", fontsize=12, color="#4a4a4a")
    plt.ylabel("Accumulated Events / Number of Seen Objects", fontsize=12, color="#4a4a4a")
    plt.xlim([0, 1])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=False, loc="upper left", ncol=2)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8c8c8c")
    ax.spines["bottom"].set_color("#8c8c8c")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    plt.show()


def plot_gt_reassoc_curve(scene_results, num_grid_points=1000, save_path=None,
                          title="ScanNet: GT Reappearance & Successful Reassociation"):
    grid = np.linspace(0, 1, num_grid_points)
    plt.figure(figsize=(10, 6))

    all_interp_reappear = []
    all_interp_success = []

    for result in scene_results.values():
        _, _, reappear_curve, success_curve, _ = _build_gt_curves_from_scene_result(result)
        if reappear_curve is None:
            continue
        all_interp_reappear.append(_interp_curve(reappear_curve, num_grid_points))
        all_interp_success.append(_interp_curve(success_curve, num_grid_points))

    if not all_interp_reappear:
        print("No valid GT-side reassociation sequences for plotting.")
        return

    mean_reappear = np.mean(all_interp_reappear, axis=0)
    mean_success = np.mean(all_interp_success, axis=0)

    for curve in all_interp_reappear:
        plt.plot(grid, curve, color="#A78C8F", alpha=0.03, linewidth=0.5, zorder=1)
    for curve in all_interp_success:
        plt.plot(grid, curve, color="#6470A6", alpha=0.03, linewidth=0.5, zorder=1)

    plt.plot(grid, mean_reappear, label="Avg GT Reappearances", color="#C97B84", linewidth=3, zorder=10,
             path_effects=[pe.Stroke(linewidth=6, foreground="white"), pe.Normal()])
    plt.plot(grid, mean_success, label="Avg Successful Reassociations", color="#6577D9", linewidth=3, zorder=10,
             path_effects=[pe.Stroke(linewidth=6, foreground="white"), pe.Normal()])

    plt.title(title, fontsize=14, fontweight="bold", color="#4a4a4a")
    plt.xlabel("Normalized Time ($t/T$)", fontsize=12, color="#4a4a4a")
    plt.ylabel("Accumulated Events / Number of Seen GT Objects", fontsize=12, color="#4a4a4a")
    plt.xlim([0, 1])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=False, loc="upper left", ncol=2)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8c8c8c")
    ax.spines["bottom"].set_color("#8c8c8c")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    plt.show()


def plot_gt_discovery_curve(scene_results, num_grid_points=1000, save_path=None,
                            title="ScanNet: GT Discovery & Successful Discovery"):
    grid = np.linspace(0, 1, num_grid_points)
    plt.figure(figsize=(10, 6))

    all_interp_disc = []
    all_interp_success = []

    for result in scene_results.values():
        disc_curve, success_curve, _, _, _ = _build_gt_curves_from_scene_result(result)
        if disc_curve is None:
            continue
        all_interp_disc.append(_interp_curve(disc_curve, num_grid_points))
        all_interp_success.append(_interp_curve(success_curve, num_grid_points))

    if not all_interp_disc:
        print("No valid GT-side discovery sequences for plotting.")
        return

    mean_disc = np.mean(all_interp_disc, axis=0)
    mean_success = np.mean(all_interp_success, axis=0)

    for curve in all_interp_disc:
        plt.plot(grid, curve, color="#90CEC6", alpha=0.03, linewidth=0.5, zorder=1)
    for curve in all_interp_success:
        plt.plot(grid, curve, color="#9088FF", alpha=0.03, linewidth=0.5, zorder=1)

    plt.plot(grid, mean_disc, label="Avg GT Discoveries", color="#6AD9CE", linewidth=3, zorder=10,
             path_effects=[pe.Stroke(linewidth=6, foreground="white"), pe.Normal()])
    plt.plot(grid, mean_success, label="Avg Successful Discoveries", color="#7A6DFF", linewidth=3, zorder=10,
             path_effects=[pe.Stroke(linewidth=6, foreground="white"), pe.Normal()])

    plt.title(title, fontsize=14, fontweight="bold", color="#4a4a4a")
    plt.xlabel("Normalized Time ($t/T$)", fontsize=12, color="#4a4a4a")
    plt.ylabel("Accumulated Events / Number of Seen GT Objects", fontsize=12, color="#4a4a4a")
    plt.xlim([0, 1])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=False, loc="upper left", ncol=2)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8c8c8c")
    ax.spines["bottom"].set_color("#8c8c8c")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    plt.show()


def write_events_csv(path, rows):
    fieldnames = [
        "scene_id", "event_type", "eval_t", "frame_idx", "gid", "gap_len",
        "window_end_eval_t", "window_end_frame_idx",
        "support_thr", "coverage_thr", "strict_ownership", "event_window",
        "supporting_pids_within_window",
        "first_success_eval_t", "first_success_frame_idx",
        "first_success_pids", "first_success_support_scores",
        "first_success_joint_coverage",
        "delay_eval",
        "best_joint_coverage", "best_joint_coverage_eval_t", "best_joint_coverage_frame_idx",
        "best_supporting_pids", "best_support_scores",
        "window_frame_records",
        "matched_pids_at_event", "successful_pids", "winning_pid",
        "successful_discovery", "successful_reassoc",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            for key, empty in [
                ("supporting_pids_within_window", []),
                ("first_success_pids", []),
                ("first_success_support_scores", {}),
                ("best_supporting_pids", []),
                ("best_support_scores", {}),
                ("window_frame_records", []),
                ("matched_pids_at_event", []),
                ("successful_pids", []),
            ]:
                row[key] = json.dumps(row.get(key, empty))
            writer.writerow(row)


def infer_scene_dir(eval_root, scene_id):
    direct = os.path.join(eval_root, scene_id)
    if os.path.isdir(direct):
        return scene_id

    candidates = [
        d for d in os.listdir(eval_root)
        if os.path.isdir(os.path.join(eval_root, d)) and d.endswith(scene_id)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidates = sorted(candidates, key=len)
        return candidates[0]
    return scene_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default="/path/to/scannet/gt")
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--scene_names", type=str, nargs="+", default=None)
    parser.add_argument("--ignore_label", type=int, default=0)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--ios_thr", type=float, default=0.5)
    parser.add_argument("--min_gap", type=int, default=1)
    parser.add_argument("--discovery_window", type=int, default=2)
    parser.add_argument("--reassoc_window", type=int, default=2)
    parser.add_argument("--support_thr", type=float, default=0.5)
    parser.add_argument("--coverage_thr", type=float, default=0.5)
    parser.add_argument("--strict_ownership", action="store_true",
                        help="If set, a supporting prediction id can only support the GT it is globally assigned to.")
    parser.add_argument("--out_dir", type=str, default="./identity_event_analysis_v6")
    parser.add_argument("--save_pred_plot", action="store_true")
    parser.add_argument("--save_gt_reassoc_plot", action="store_true")
    parser.add_argument("--save_gt_discovery_plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = ScannetVOSDataset(gt_dir=args.gt_dir, pred_dir=args.pred_root)

    scene_ids = args.scene_names if args.scene_names else sorted([s for s in dataset.get_scene_ids() if s.endswith("_00")])

    all_scene_results = {}
    csv_rows = []

    for scene_id in scene_ids:
        actual_scene_dir = infer_scene_dir(args.pred_root, scene_id)
        preds, gts, frame_indices = dataset.load_scene_frames(actual_scene_dir)
        if len(preds) == 0 or len(gts) == 0:
            print(f"Skipping {scene_id}: missing predictions or GT.")
            continue

        result = analyze_scene_identity_events(
            preds=preds,
            gts=gts,
            frame_indices=frame_indices,
            ignore_label=args.ignore_label,
            iou_thr=args.iou_thr,
            ios_thr=args.ios_thr,
            min_gap=args.min_gap,
            discovery_window=args.discovery_window,
            reassoc_window=args.reassoc_window,
            support_thr=args.support_thr,
            coverage_thr=args.coverage_thr,
            strict_ownership=args.strict_ownership,
        )

        all_scene_results[scene_id] = result
        for ev in result["events"]:
            row = dict(ev)
            row["scene_id"] = scene_id
            csv_rows.append(row)

        print(f"[{scene_id}] "
              f"disc={result['summary']['gt_discovery_events']} | "
              f"disc_succ={result['summary']['successful_discoveries']} | "
              f"reapp={result['summary']['gt_reappearance_events']} | "
              f"reassoc={result['summary']['successful_reassociations']}")

    summary = {
        "num_scenes": len(all_scene_results),
        "total_gt_discovery_events": int(sum(v["summary"]["gt_discovery_events"] for v in all_scene_results.values())),
        "total_successful_discoveries": int(sum(v["summary"]["successful_discoveries"] for v in all_scene_results.values())),
        "total_failed_discoveries": int(sum(v["summary"]["failed_discoveries"] for v in all_scene_results.values())),
        "total_gt_reappearance_events": int(sum(v["summary"]["gt_reappearance_events"] for v in all_scene_results.values())),
        "total_successful_reassociations": int(sum(v["summary"]["successful_reassociations"] for v in all_scene_results.values())),
        "total_failed_reassociations": int(sum(v["summary"]["failed_reassociations"] for v in all_scene_results.values())),
        "discovery_window": int(args.discovery_window),
        "reassoc_window": int(args.reassoc_window),
        "event_support_thr": float(args.support_thr),
        "event_coverage_thr": float(args.coverage_thr),
        "strict_ownership": bool(args.strict_ownership),
    }

    total_disc = summary["total_gt_discovery_events"]
    total_reapp = summary["total_gt_reappearance_events"]
    summary["overall_discovery_success_rate"] = float(summary["total_successful_discoveries"] / total_disc) if total_disc > 0 else 0.0
    summary["overall_reassoc_success_rate"] = float(summary["total_successful_reassociations"] / total_reapp) if total_reapp > 0 else 0.0

    with open(os.path.join(args.out_dir, "identity_event_summary.json"), "w") as f:
        json.dump({"global_summary": summary, "per_scene": {k: v["summary"] for k, v in all_scene_results.items()}}, f, indent=2)

    with open(os.path.join(args.out_dir, "identity_events.json"), "w") as f:
        json.dump(all_scene_results, f, indent=2)

    with open(os.path.join(args.out_dir, "prediction_side_frame_ids.json"), "w") as f:
        json.dump({k: v["prediction_side_frame_ids"] for k, v in all_scene_results.items()}, f)

    write_events_csv(os.path.join(args.out_dir, "identity_events.csv"), csv_rows)

    print("\n[Done]")
    print(json.dumps(summary, indent=2))

    pred_plot_path = os.path.join(args.out_dir, "pred_reappearance_curve.png") if args.save_pred_plot else None
    gt_reassoc_plot_path = os.path.join(args.out_dir, "gt_reassoc_curve.png") if args.save_gt_reassoc_plot else None
    gt_discovery_plot_path = os.path.join(args.out_dir, "gt_discovery_curve.png") if args.save_gt_discovery_plot else None

    plot_prediction_reappearance_curve(all_scene_results, background_id=args.ignore_label, save_path=pred_plot_path)
    plot_gt_reassoc_curve(all_scene_results, save_path=gt_reassoc_plot_path)
    plot_gt_discovery_curve(all_scene_results, save_path=gt_discovery_plot_path)


if __name__ == "__main__":
    main()
