import math
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def _compute_mask_compactness(mask: torch.Tensor) -> float:
    """
    Compactness = 4π × area / perimeter²
    Circle = 1.0, elongated/thin shapes → 0.0

    Perimeter approximated by counting boundary pixels —
    pixels that are True but have at least one False 4-neighbour.
    Pure tensor ops, no scipy needed.
    """
    import torch.nn.functional as F

    if not mask.any():
        return 0.0

    area = mask.float().sum().item()

    # Detect boundary pixels via max-pooling erosion.
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    # A pixel survives erosion only if all 4-neighbours are also true.
    eroded = F.max_pool2d(
        1.0 - mask_f, kernel_size=3, stride=1, padding=1
    )
    eroded = (eroded == 0).squeeze()     # True where interior (all neighbours True)
    boundary = mask & (~eroded)          # boundary = mask minus interior
    perimeter = boundary.float().sum().item()

    if perimeter == 0:
        return 1.0  # single pixel or fully solid — treat as compact

    import math
    return (4 * math.pi * area) / (perimeter ** 2)

def _is_in_border_margin(mask: torch.Tensor, margin_ratio: float = 0.05) -> bool:
    """
    Delays mask registration if its center of mass is within the outer X% of the frame.
    Prevents picking up partial/fragmented objects as they enter or exit the camera view.
    """
    H, W = mask.shape
    ys, xs = torch.where(mask)
    
    if len(ys) == 0:
        return False
        
    # Calculate the centroid.
    cy = ys.float().mean().item()
    cx = xs.float().mean().item()
    
    margin_y = H * margin_ratio
    margin_x = W * margin_ratio
    
    # Check whether the centroid falls in the outer band.
    if cy < margin_y or cy > (H - margin_y):
        return True
    if cx < margin_x or cx > (W - margin_x):
        return True
        
    return False

def _compute_bbox_aspect_ratio(mask: torch.Tensor) -> float:
    """
    min(height, width) / max(height, width) of the mask bounding box.
    Square = 1.0, thin sliver → 0.0
    """
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return 0.0
    h = (ys.max() - ys.min()).item() + 1
    w = (xs.max() - xs.min()).item() + 1
    return min(h, w) / max(h, w)

def _is_sliver(mask: torch.Tensor,
               min_compactness: float = 0.03,
               min_aspect_ratio: float = 0.04) -> bool:
    """
    Returns True if the mask is a sliver and should be dropped.
    Fails either compactness OR aspect ratio test → sliver.
    """
    if _compute_mask_compactness(mask) < min_compactness:
        return True
    if _compute_bbox_aspect_ratio(mask) < min_aspect_ratio:
        return True
    return False

def load_external_masks_per_frame(
    frame_indices,
    npy_dir,
    inference_state,
    device="cpu",
    min_area_ratio=0.005,    # minimum mask area as fraction of total image area (0.1%)
    min_bbox_occupancy=0.30, # minimum ratio of mask area to its bounding box area (30%)
):
    """
    Load and filter external segmentation index maps, resized to video resolution.

    Filtering criteria (both must pass):
      1. Area ratio:      mask_area / (H * W) >= min_area_ratio
                          Default 0.001 (0.1%) — removes tiny specks.
                          Typical range: 0.0008 (COCO small) to 0.005 (conservative).
      2. BBox occupancy:  mask_area / bbox_area >= min_bbox_occupancy
                          Default 0.30 (30%) — removes sparse/irregular masks like
                          wires, text, thin structures that are hard to track.

    .npy format:
        shape:  [..., H_orig, W_orig]
        values: -1=background, 0,1,2,...=mask indices
    After raw[-1] + 1:
        0=background, 1,2,3,...=masks (1-based, as expected by new_mask_fn)

    Returns:
        dict {frame_idx: Tensor[H, W] int64}, 0=background, 1..M=surviving masks
        Surviving masks are re-indexed densely (no gaps in indices).
    """
    video_H = inference_state["video_height"]
    video_W = inference_state["video_width"]
    image_area = video_H * video_W
    external_masks_per_frame = {}

    for frame_idx in frame_indices:
        npy_path = os.path.join(npy_dir, f"{frame_idx:06d}_segmaps.npy")
        if not os.path.exists(npy_path):
            continue

        raw = np.load(npy_path)       # [..., H_orig, W_orig]
        index_map = raw[-1] + 1       # [H_orig, W_orig], 0=bg, 1..M=masks

        orig_H, orig_W = index_map.shape

        # Resize to video resolution using nearest-neighbour (preserves label IDs)
        index_map_tensor = torch.from_numpy(index_map.astype(np.int64))
        if orig_H != video_H or orig_W != video_W:
            index_map_tensor = F.interpolate(
                index_map_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=(video_H, video_W),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()

        # Filter masks.
        mask_indices = index_map_tensor.unique()
        mask_indices = mask_indices[mask_indices > 0]  # drop background

        if mask_indices.numel() == 0:
            continue

        # Build filtered output map, re-indexing densely from 1
        filtered_map = torch.zeros(video_H, video_W, dtype=torch.long)
        new_idx = 1

        for idx in mask_indices:
            binary = (index_map_tensor == idx)  # [H, W] bool

            # Area ratio filter.
            mask_area = binary.sum().item()
            if mask_area / image_area < min_area_ratio:
                continue

            # Bounding-box occupancy filter.
            ys, xs = torch.where(binary)
            y_min, y_max = ys.min().item(), ys.max().item()
            x_min, x_max = xs.min().item(), xs.max().item()
            bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
            if bbox_area == 0 or mask_area / bbox_area < min_bbox_occupancy:
                continue

            filtered_map[binary] = new_idx
            new_idx += 1

        if new_idx == 1:
            continue  # all masks filtered out for this frame

        external_masks_per_frame[frame_idx] = filtered_map.to(device)

    return external_masks_per_frame

def _find_contained_parts(
    whole_id: int,
    frame_idx: int,
    established_ids: List[int],
    all_results: dict,
    containment_threshold: float = 0.7,
    min_area_ratio: float = 1.5,
    bbox_iou_threshold: float = 0.1,
    bbox_ios_threshold: float = 0.3,    # intersection over smaller bbox
) -> List[int]:
    """
    Find established 'part' objects whose masks are largely contained
    within the newly-promoted 'whole' mask.

    Filter cascade (cheapest to most expensive):
      1. Sort candidates by mask area descending
      2. Area ratio gate:   whole_area / part_area >= min_area_ratio
      3. Bbox proximity:    bbox_iou >= bbox_iou_threshold
                         OR bbox_ios (intersection / part_bbox_area) >= bbox_ios_threshold
         Purpose: cheaply reject spatially disjoint masks.
         Contour masks naturally pass (their bbox ≈ whole bbox → high bbox_iou).
         Compact parts pass via bbox_iou.
      4. Pixel mask IoS:    intersection(whole, part) / part_area >= containment_threshold
    """
    frame_result = all_results.get(frame_idx, {})
    whole_mask = frame_result.get(whole_id)
    if whole_mask is None:
        return []

    whole_bin = torch.as_tensor(np.squeeze(whole_mask), dtype=torch.bool)
    if not whole_bin.any():
        return []

    whole_area = whole_bin.sum().item()

    # Compute the whole-object bbox once.
    whole_ys, whole_xs = torch.where(whole_bin)
    wy0, wy1 = whole_ys.min().item(), whole_ys.max().item()
    wx0, wx1 = whole_xs.min().item(), whole_xs.max().item()
    whole_bbox_area = (wy1 - wy0 + 1) * (wx1 - wx0 + 1)

    # Collect candidate parts and sort by area, largest first.
    candidates = []
    for part_id in established_ids:
        part_mask = frame_result.get(part_id)
        if part_mask is None:
            continue
        part_bin = torch.as_tensor(np.squeeze(part_mask), dtype=torch.bool)
        if not part_bin.any():
            continue
        part_area = part_bin.sum().item()
        candidates.append((part_id, part_bin, part_area))

    candidates.sort(key=lambda x: x[2], reverse=True)

    parts_to_absorb = []

    for part_id, part_bin, part_area in candidates:

        # Area-ratio gate.
        if whole_area / part_area < min_area_ratio:
            continue  # part too large relative to whole — not a part-whole pair

        # Bbox proximity gate rejects spatially disjoint masks before pixel checks.
        part_ys, part_xs = torch.where(part_bin)
        py0, py1 = part_ys.min().item(), part_ys.max().item()
        px0, px1 = part_xs.min().item(), part_xs.max().item()
        part_bbox_area = (py1 - py0 + 1) * (px1 - px0 + 1)

        # Bbox intersection.
        inter_y0 = max(wy0, py0)
        inter_y1 = min(wy1, py1)
        inter_x0 = max(wx0, px0)
        inter_x1 = min(wx1, px1)

        if inter_y1 < inter_y0 or inter_x1 < inter_x0:
            continue  # bboxes don't even overlap — skip immediately

        bbox_inter_area = (inter_y1 - inter_y0 + 1) * (inter_x1 - inter_x0 + 1)
        bbox_union_area = whole_bbox_area + part_bbox_area - bbox_inter_area
        bbox_iou = bbox_inter_area / bbox_union_area
        bbox_ios = bbox_inter_area / part_bbox_area  # intersection over smaller (part) bbox

        if bbox_iou < bbox_iou_threshold and bbox_ios < bbox_ios_threshold:
            continue  # spatially too disjoint — skip pixel check

        # Pixel-level mask IoS.
        intersection = (whole_bin & part_bin).sum().item()
        containment = intersection / part_area

        if containment >= containment_threshold:
            parts_to_absorb.append(part_id)

    return parts_to_absorb

def mask_merge(masks_1, masks_2, min_ratio=0.1):
    masks_all = list(masks_1) + list(masks_2)
    N = len(masks_all)
    if N == 0:
        return [], np.full((0, 0), -1, dtype=np.int32)

    m_stack = np.stack([np.asarray(m["masks"]).astype(bool) for m in masks_all])
    scores = np.array([m["quality_score"] for m in masks_all], dtype=np.float32)
    logits = m_stack * scores[:, None, None]
    
    seg_w = np.argmax(logits, axis=0)
    seg_w[~m_stack.any(axis=0)] = -1

    orig_areas = m_stack.sum(axis=(1, 2))
    assigned_areas = np.bincount(seg_w[seg_w >= 0], minlength=N)
    keep_mask = (assigned_areas >= (min_ratio * orig_areas)) & (assigned_areas > 0)
    
    invalid_indices = (seg_w >= 0) & (~keep_mask[seg_w])
    seg_w[invalid_indices] = -1

    winner_indices = np.where(keep_mask)[0]
    winner_masks = []
    for idx in winner_indices:
        ann = dict(masks_all[idx])
        ann["masks"] = (seg_w == idx)
        ann["area"] = int(assigned_areas[idx])
        winner_masks.append(ann)

    return winner_masks, seg_w

def hierarchical_mask_merge(segmaps, survival_ratio=0.1):
    mask_s, mask_m, mask_l = segmaps
    
    # Avoid stacking empty mask lists.
    if not mask_s and not mask_m and not mask_l:
        return np.array([])

    # Pass 1: small vs. medium masks.
    if mask_s and mask_m:
        merged_sm, _ = mask_merge(mask_s, mask_m, survival_ratio)
    else:
        merged_sm = mask_s if mask_s else mask_m
    
    # Pass 2: merged small/medium masks vs. large masks.
    if merged_sm and mask_l:
        _, seg_l = mask_merge(merged_sm, mask_l, survival_ratio)
    elif merged_sm:
        seg_l = mask2segmap(merged_sm)
    else:
        seg_l = mask2segmap(mask_l)

    return seg_l

def mask2segmap(masks):
    if not masks:
        return np.zeros((0, 0), dtype=np.int32)
    m_stack = np.stack([np.asarray(m["masks"]).astype(bool) for m in masks])
    
    areas = np.array([
        m.get("area", m_stack[i].sum()) 
        for i, m in enumerate(masks)
    ], dtype=np.float32)

    weighted_masks = m_stack * areas[:, None, None]
    seg_map = np.argmax(weighted_masks, axis=0)
    seg_map[~m_stack.any(axis=0)] = -1
    
    return seg_map.astype(np.int32)
