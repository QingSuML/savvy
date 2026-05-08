from contextlib import contextmanager
from typing import Dict, Set
from unittest.mock import patch

import numpy as np
import torch

from .mask_processing import _is_in_border_margin, _is_sliver


@contextmanager
def _suppress_tqdm():
    """Suppress the internal tqdm bar inside propagate_in_video."""
    with patch("sam2.sam2_video_predictor.tqdm", lambda x, **kwargs: x):
        yield

def _bootstrap_initial_frame(predictor, inference_state, index_map, start_frame_idx, next_obj_id):
    """
    Register all masks in index_map at start_frame_idx unconditionally.
    index_map: [H, W] int64, 0=background, 1..M=masks
    """
    device = inference_state["device"]
    index_map = index_map.to(device)
    mask_indices = index_map.unique()
    mask_indices = mask_indices[mask_indices > 0]  # drop background

    for idx in mask_indices:
        binary_mask = (index_map == idx)           # [H, W] bool
        if binary_mask.sum() == 0:
            continue
        predictor.add_new_mask(
            inference_state,
            frame_idx=start_frame_idx,
            obj_id=next_obj_id,
            mask=binary_mask,
        )
        next_obj_id += 1

    _flush_new_obj_temp_outputs(predictor, inference_state)
    return next_obj_id

def _flush_new_obj_temp_outputs(predictor, inference_state):
    """
    Replicate the memory-encoder step of propagate_in_video_preflight,
    but only for objects that still have entries in temp_output_dict_per_obj.
    Safe to call mid-propagation since existing cond_frame_outputs are untouched.
    """
    batch_size = predictor._get_obj_num(inference_state)
    for obj_idx in range(batch_size):
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]

        for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
            for frame_idx, out in list(obj_temp_output_dict[storage_key].items()):
                if out["maskmem_features"] is None:
                    high_res_masks = torch.nn.functional.interpolate(
                        out["pred_masks"].to(inference_state["device"]),
                        size=(predictor.image_size, predictor.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    maskmem_features, maskmem_pos_enc = predictor._run_memory_encoder(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        batch_size=1,
                        high_res_masks=high_res_masks,
                        object_score_logits=out["object_score_logits"],
                        is_mask_from_pts=True,
                    )
                    out["maskmem_features"] = maskmem_features
                    out["maskmem_pos_enc"] = maskmem_pos_enc

                # Promote staged outputs into the persistent output store.
                obj_output_dict[storage_key][frame_idx] = out
                # A conditioning frame supersedes any non-conditioning output at the same frame.
                if storage_key == "cond_frame_outputs":
                    obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

            obj_temp_output_dict[storage_key].clear()

def apply_seniority_suppression_to_frame(
    frame_masks: Dict[int, np.ndarray],
    established_ids: set,
    consistency_scores: Dict[int, float],
    continuous_presence_streaks: Dict[int, int],
    iou_threshold: float = 0.5,
    margin: float = 0.05,
) -> Set[int]:
    """
    Resolve overlaps among established objects on a single frame.

    Decision tree for each overlapping pair (A, B) with IoU >= iou_threshold:

      1. Clear win  — |score_A - score_B| >= margin
                      → higher score wins, lower score suppressed

      2. Ambiguous  — |score_A - score_B| < margin
                      → longer continuous presence streak wins
                      → shorter streak suppressed (defer to stable object)

      3. Full tie   — equal streaks (or no scores available)
                      → seniority: lower obj_id wins

    Args:
        frame_masks:                {obj_id: mask (H,W) bool}
        established_ids:            set of established obj_ids to consider
        consistency_scores:         {obj_id: float} pre-computed self-consistency scores
        continuous_presence_streaks:{obj_id: int} frames since last (re)appearance
        iou_threshold:              minimum IoU to trigger suppression
        margin:                     minimum score gap to constitute a clear win

    Returns:
        Set of suppressed obj_ids
    """
    est_present = {
        oid: frame_masks[oid].squeeze()
        for oid in established_ids
        if frame_masks.get(oid) is not None
        and frame_masks[oid].squeeze().any()
    }

    if len(est_present) <= 1:
        return set()

    suppressed_ids = set()
    oids = list(est_present.keys())

    for i in range(len(oids)):
        a = oids[i]
        if a in suppressed_ids:
            continue
        for j in range(i + 1, len(oids)):
            b = oids[j]
            if b in suppressed_ids:
                continue

            intersection = (est_present[a] & est_present[b]).sum()
            if intersection == 0:
                continue

            union = (est_present[a] | est_present[b]).sum()
            iou = intersection / union if union > 0 else 0.0

            if iou < iou_threshold:
                continue

            score_a = consistency_scores.get(a, None)
            score_b = consistency_scores.get(b, None)

            if score_a is not None and score_b is not None:
                diff = abs(score_a - score_b)
                if diff >= margin:
                    # Clear win: higher self-consistency score takes precedence.
                    loser = b if score_a >= score_b else a
                else:
                    # Ambiguous pair: prefer the object with longer continuous presence.
                    streak_a = continuous_presence_streaks.get(a, 0)
                    streak_b = continuous_presence_streaks.get(b, 0)
                    if streak_a != streak_b:
                        loser = b if streak_a >= streak_b else a
                    else:
                        # Full tie: prefer the older object ID.
                        loser = max(a, b)
            else:
                # Without scores, fall back to seniority.
                loser = max(a, b)

            suppressed_ids.add(loser)
            frame_masks.pop(loser, None)

    return suppressed_ids

def new_mask_fn(
    frame_idx,
    video_res_masks,
    external_masks,
    inference_state,
    next_obj_id,
    survival_threshold=0.1,
    extra_suppress_masks=None,
):
    device = inference_state["device"]
    external_masks = external_masks.to(device)
    H, W = external_masks.shape

    mask_indices = external_masks.unique()
    mask_indices = mask_indices[mask_indices > 0]
    num_external = mask_indices.shape[0]

    if num_external == 0:
        return [], next_obj_id

    num_tracked = video_res_masks.shape[0]

    ext_channels = torch.stack([
        (external_masks == idx).float() * (rank + 1.0)
        for rank, idx in enumerate(mask_indices)
    ], dim=0)  # [M, H, W]

    tracked_binary = (video_res_masks[:, 0, :, :] > 0).float()
    tracked_scored = tracked_binary * torch.arange(
        num_external + 1, num_external + 1 + num_tracked, device=device
    ).float()[:, None, None]

    # Extra suppression masks act as soft veto regions between external and tracked masks.
    suppress_channels = []
    if extra_suppress_masks:
        for i, smask in enumerate(extra_suppress_masks):
            smask_t = torch.as_tensor(np.squeeze(smask), dtype=torch.float32, device=device)
            if smask_t.shape != (H, W):
                smask_t = torch.nn.functional.interpolate(
                    smask_t.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="nearest"
                ).squeeze()
            # Suppression masks win over external candidates but not tracked masks.
            suppress_channels.append(
                smask_t * (num_external + 0.5 + i * 0.01)
            )

    suppress_tensor = torch.stack(suppress_channels, dim=0) \
                      if suppress_channels else torch.zeros(0, H, W, device=device)

    bg = torch.full((1, H, W), 0.1, device=device)
    scored = torch.cat([
        bg,
        ext_channels,
        suppress_tensor.to(device) if suppress_tensor.numel() > 0
            else torch.zeros(0, H, W, device=device),
        tracked_scored,
    ], dim=0)
    hard_mask = torch.argmax(scored, dim=0)

    new_objects = []
    for rank, idx in enumerate(mask_indices):
        label_in_hard = rank + 1
        original_area = (external_masks == idx).sum().item()
        if original_area == 0:
            continue
        surviving_area = (hard_mask == label_in_hard).sum().item()
        if surviving_area == 0:
            continue
        if surviving_area / original_area < survival_threshold:
            continue

        surviving_mask = (hard_mask == label_in_hard)

        if _is_sliver(surviving_mask.cpu()):
            continue

        # Delay partial objects that are entering or leaving at the image border.
        if _is_in_border_margin(surviving_mask.cpu(), margin_ratio=0.05):
            continue

        new_objects.append((next_obj_id, surviving_mask))
        next_obj_id += 1

    return new_objects, next_obj_id

def _collect_obj_ptrs_and_masks(inference_state, obj_ids, frame_idx):
    """
    Retrieve obj_ptr, pred_masks, and object_score_logits for each obj_id
    at frame_idx from output_dict_per_obj.
    """
    obj_id_to_idx = inference_state["obj_id_to_idx"]
    obj_ptrs           = {}
    pred_masks         = {}
    object_score_logits = {}

    for obj_id in obj_ids:
        obj_idx = obj_id_to_idx.get(obj_id)
        if obj_idx is None:
            continue
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]

        out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        if out is None:
            out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
        if out is None:
            continue
        
        obj_ptrs[obj_id]            = out["obj_ptr"].detach().cpu()             # [1, hidden_dim]
        pred_masks[obj_id]          = out["pred_masks"]          # [1, 1, H, W]
        object_score_logits[obj_id] = out["object_score_logits"].item()  # scalar

    return obj_ptrs, pred_masks, object_score_logits

def _prune_old_non_cond_outputs(inference_state, current_frame_idx, num_maskmem, stride):
    """
    Drop non_cond_frame_outputs entries that are outside the memory attention window.
    SAM2 only looks back num_maskmem frames with the given stride, so anything older
    than (current_frame_idx - num_maskmem * stride) is never accessed again.

    This is the single biggest memory saving for long videos.
    """
    cutoff = current_frame_idx - num_maskmem * stride - 1  # keep a small buffer

    for obj_output_dict in inference_state["output_dict_per_obj"].values():
        non_cond = obj_output_dict["non_cond_frame_outputs"]
        stale_keys = [t for t in non_cond if t < cutoff]
        for t in stale_keys:
            del non_cond[t]

def _strip_pred_masks_from_old_frames(inference_state, current_frame_idx, keep_last_n=10):
    """
    Set pred_masks=None for non_cond_frame_outputs older than keep_last_n frames.
    """
    cutoff = current_frame_idx - keep_last_n

    for obj_output_dict in inference_state["output_dict_per_obj"].values():
        non_cond = obj_output_dict["non_cond_frame_outputs"]
        for t, out in non_cond.items():
            if t < cutoff and out.get("pred_masks") is not None:
                out["pred_masks"] = None

def backfill_transient_gaps(all_results, transient_mask_buffer):
    """
    Backfill suppressed transient masks into all_results using the
    video-resolution masks captured during propagation.

    For promoted objects: fills all frames from entry to promotion.
    For merged objects: the remap loop already handled it — buffer is
    just cleaned up.

    Modifies all_results in-place.
    """
    for obj_id, frame_masks in transient_mask_buffer.items():
        for frame_idx, mask in frame_masks.items():
            if frame_idx not in all_results:
                continue
            if obj_id in all_results[frame_idx]:
                continue
            if not mask.any():
                continue
            all_results[frame_idx][obj_id] = mask

    return all_results

def _merge_transient_into_established(
    predictor,
    inference_state,
    transient_obj_id: int,
    established_obj_id: int,
):
    """
    Merge a transient track into an established one.

    Strategy: keep the established track's entire memory intact (it has longer,
    more reliable history). Simply remove the transient object entirely.
    The established object's propagation continues uninterrupted.

    Uses predictor.remove_object which correctly re-indexes all state dicts.
    """
    predictor.remove_object(
        inference_state,
        obj_id=transient_obj_id,
        strict=False,
        need_output=False,
    )
