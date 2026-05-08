import random

import cv2
import numpy as np


def apply_temporal_dropout(seg_list, drop_prob=0.3, ignore_label=0):
    """
    Randomly converts predictions to void to stress spatial continuity.
    Unlike ID switching, this does not create new IDs.
    """
    if not seg_list: return []
    
    dropped_seg_list = []
    
    for seg in seg_list:
        seg_np = np.array(seg).copy()
        ids_in_frame = np.unique(seg_np)
        
        for obj_id in ids_in_frame:
            if obj_id == ignore_label:
                continue
                
            if random.random() < drop_prob:
                seg_np[seg_np == obj_id] = ignore_label
                
        dropped_seg_list.append(seg_np)
        
    return dropped_seg_list

def apply_macro_sever(seg_list, num_severs=1, min_track_length=30, ignore_label=0):
    """
    Slices a continuous track into large, distinct ID chunks to expose 
    the temporal blindness of sliding-window VPQ.
    """
    if not seg_list: return []
    
    # Map object lifespans.
    id_lifespans = {}
    for t, seg in enumerate(seg_list):
        for obj_id in np.unique(np.array(seg)):
            if obj_id == ignore_label: continue
            if obj_id not in id_lifespans:
                id_lifespans[obj_id] = [t, t]
            else:
                id_lifespans[obj_id][1] = t
                
    global_max_id = max([np.max(np.array(s)) for s in seg_list] + [0])
    
    # Schedule track severing events.
    sever_map = {t: {} for t in range(len(seg_list))}
    
    for obj_id, (start, end) in id_lifespans.items():
        track_length = end - start + 1
        if track_length >= min_track_length:
            # Cut the track into equal fractions.
            step = track_length // (num_severs + 1)
            
            for i in range(1, num_severs + 1):
                sever_t = start + (step * i)
                global_max_id += 1
                sever_map[sever_t][obj_id] = global_max_id
                # Subsequent severs apply to the new tail ID.
                obj_id = global_max_id 

    # Apply scheduled remaps to the sequence.
    severed_seg_list = []
    current_remaps = {}
    
    for t, seg in enumerate(seg_list):
        seg_np = np.array(seg).copy()
        
        if t in sever_map:
            for old_id, new_id in sever_map[t].items():
                current_remaps[old_id] = new_id
                
        # Active remaps persist after their sever frame.
        for old_id, new_id in current_remaps.items():
             seg_np[seg_np == old_id] = new_id
             
        severed_seg_list.append(seg_np)
        
    return severed_seg_list

def apply_sparse_id_flickering(
    seg_list,
    noise_fraction=0.05,
    min_track_length=8,
    min_gap=2,
    ignore_label=0,
    seed=0,
):
    """
    Simulates sparse temporal identity flicker by replacing an object's ID
    with a fresh hallucinated ID on a small number of isolated frames.

    This preserves spatial masks and only perturbs identity assignment.

    Args:
        seg_list: list of segmentation maps
        noise_fraction: fraction of frames in each eligible track to flicker
        min_track_length: minimum lifespan for a track to be eligible
        min_gap: minimum frame gap between flickers for the same track
        ignore_label: void/background label
        seed: RNG seed
    """
    if not seg_list:
        return []

    rng = random.Random(seed)

    # Map frames where each ID appears.
    id_to_frames = {}
    for t, seg in enumerate(seg_list):
        seg_np = np.array(seg)
        for obj_id in np.unique(seg_np):
            if obj_id == ignore_label:
                continue
            id_to_frames.setdefault(int(obj_id), []).append(t)

    global_max_id = max([int(np.max(np.array(seg))) for seg in seg_list] + [0])

    flicker_map = {t: {} for t in range(len(seg_list))}

    for obj_id, frames in id_to_frames.items():
        track_length = len(frames)
        if track_length < min_track_length:
            continue

        n_flickers = max(1, int(round(track_length * noise_fraction)))
        candidate_frames = frames[:]
        rng.shuffle(candidate_frames)

        chosen = []
        for t in candidate_frames:
            # Enforce sparse isolated flickers.
            if all(abs(t - prev_t) > min_gap for prev_t in chosen):
                chosen.append(t)
                if len(chosen) >= n_flickers:
                    break

        for t in sorted(chosen):
            global_max_id += 1
            flicker_map[t][obj_id] = global_max_id

    # Apply flickers.
    flickered_seg_list = []
    for t, seg in enumerate(seg_list):
        seg_np = np.array(seg).copy()
        for old_id, new_id in flicker_map[t].items():
            seg_np[seg_np == old_id] = new_id
        flickered_seg_list.append(seg_np)

    return flickered_seg_list

def apply_dynamic_spatial_dilation(
    seg_list,
    base_iters=1,
    ignore_label=0,
    mode="void_only",
    random_order=False,
    seed=0,
):
    """
    Dilates each instance mask dynamically based on its area in the current frame.

    Modes:
        - "void_only": current behavior; only expands into ignore_label space.
        - "overwrite": aggressive dilation; expanded pixels directly write obj_id,
                       even over labeled regions. Strong clutter stress test.
        - "majority": aggressive dilation with per-pixel voting among competing IDs.
                      More stable than overwrite when many masks collide.

    Args:
        seg_list: list of HxW instance maps
        base_iters: base dilation severity
        ignore_label: background / void label
        mode: dilation mode
        random_order: if True, shuffle instance order per frame
        seed: RNG seed used when random_order=True
    """
    if not seg_list:
        return []

    assert mode in ["void_only", "overwrite", "majority"]

    H, W = np.array(seg_list[0]).shape[:2]
    dilated_seg_list = []

    rng = random.Random(seed)

    for frame_idx, seg in enumerate(seg_list):
        seg_np = np.array(seg).astype(np.int32)
        ids = [int(x) for x in np.unique(seg_np) if x != ignore_label]

        if random_order:
            rng.shuffle(ids)

        # Start from original segmentation
        if mode in ["void_only", "overwrite"]:
            dilated_seg = np.copy(seg_np)
        elif mode == "majority":
            # votes[y, x] = dict-like accumulation per ID
            # store as dense array [num_ids, H, W] for speed
            id_to_local = {obj_id: i for i, obj_id in enumerate(ids)}
            votes = np.zeros((len(ids), H, W), dtype=np.uint16)

        for obj_id in ids:
            mask = (seg_np == obj_id).astype(np.uint8)
            area = int(mask.sum())
            if area == 0:
                continue

            iters, k_size = _get_dynamic_dilation_params(area, H, W, base_iters)
            kernel = np.ones((k_size, k_size), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=iters).astype(bool)

            if mode == "void_only":
                expansion_zone = np.logical_and(dilated_mask, ~mask.astype(bool))
                valid_expansion = np.logical_and(expansion_zone, seg_np == ignore_label)
                dilated_seg[valid_expansion] = obj_id

            elif mode == "overwrite":
                # Strong stress test:
                # any newly covered pixel gets overwritten by this object.
                # original pixels are kept naturally because dilated_mask includes them.
                dilated_seg[dilated_mask] = obj_id

            elif mode == "majority":
                local_idx = id_to_local[obj_id]
                votes[local_idx, dilated_mask] += 1

        if mode == "majority":
            # Preserve original labeled pixels strongly by giving original owner +2 votes
            # This avoids absurd collapse while still allowing clutter expansion.
            for obj_id in ids:
                local_idx = id_to_local[obj_id]
                orig_mask = (seg_np == obj_id)
                votes[local_idx, orig_mask] += 2

            max_votes = votes.max(axis=0)
            winner_local = votes.argmax(axis=0)

            dilated_seg = np.copy(seg_np)

            # Only change pixels where some dilation voted
            active = max_votes > 0
            if np.any(active):
                local_to_id = {i: obj_id for obj_id, i in id_to_local.items()}
                winner_ids = np.vectorize(local_to_id.get)(winner_local)
                dilated_seg[active] = winner_ids[active]

        dilated_seg_list.append(dilated_seg)

    return dilated_seg_list

def _get_dynamic_dilation_params(area, H, W, base_iters=1):
    """
    Dynamically scales the kernel size and iterations based on the object's size
    relative to a standard 640x480 resolution.
    """
    scale_factor = (H * W) / (640 * 480)
    
    # Small objects (< 1024 scaled pixels)
    if area < 1024 * scale_factor: 
        # Very light dilation: 3x3 kernel, potentially fewer iterations
        # to prevent the object from turning into an unrecognizable circle
        return max(1, base_iters // 2), 3 
        
    # Medium objects (1024 - 9216 scaled pixels)
    elif area < 9216 * scale_factor: 
        # Standard baseline dilation
        return base_iters, 5
        
    # Large objects (> 9216 scaled pixels)
    else: 
        # Heavy dilation: 7x7 kernel and doubled iterations to ensure 
        # the boundary bleed is proportionally significant
        return base_iters * 2, 7
