from collections import defaultdict
from itertools import combinations

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _to_numpy(seg):
    if HAS_TORCH and isinstance(seg, torch.Tensor):
        return seg.detach().cpu().numpy()
    return np.array(seg) if not isinstance(seg, np.ndarray) else seg

def _get_size_label(area, H, W):
    """Adaptive sizing based on COCO reference (640x480)."""
    scale_factor = (H * W) / (640 * 480)
    if area < 1024 * scale_factor: return 'S'
    elif area < 9216 * scale_factor: return 'M'
    else: return 'L'

def _collect_global_ids(seg_list, ignore_label=None):
    id_set = set()
    for seg in seg_list:
        arr = _to_numpy(seg).astype(np.int64)
        frame_ids = np.unique(arr)
        if ignore_label is not None:
            frame_ids = frame_ids[frame_ids != ignore_label]
        id_set.update(frame_ids.tolist())
    ids = sorted(id_set)
    return ids, {sid: i for i, sid in enumerate(ids)}

def _get_frame_masks(arr, ignore_label=None):
    arr = arr.astype(np.int64)
    ids = np.unique(arr)
    if ignore_label is not None: ids = ids[ids != ignore_label]
    masks, areas = {}, {}
    for sid in ids:
        mask = (arr == sid)
        area = int(mask.sum())
        if area > 0:
            masks[sid], areas[sid] = mask, area
    return masks, areas

def _compute_frame_matches(pred_masks, 
                           pred_areas, 
                           gt_masks, 
                           gt_areas, 
                           iou_thr, 
                           ios_thr,
                           valid_gt_mask=None):
    """
    Void-tolerant frame matching:
    prediction pixels falling into GT void/unlabeled area are ignored in IoU/IoP.
    """
    frame_matches = {gid: set() for gid in gt_masks.keys()}
    frame_inters = defaultdict(dict)  # pid -> gid -> inter_area

    for pid, pmask in pred_masks.items():
        if valid_gt_mask is not None:
            pmask_valid = np.logical_and(pmask, valid_gt_mask)
        else:
            pmask_valid = pmask

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

def _compute_identity_concentration(
    seg_list,
    gt_list,
    p_ids,
    g_ids,
    p_map,
    g_map,
    ignore_label=0,
    iou_thr=0.5,
    ios_thr=0.5,
    use_void_tolerant=True,
):
    """
    Structural Identity Concentration (IC).

    Core intention:
      - video-level, structural
      - count how many distinct predictions have EVER matched each GT
      - count how many distinct GTs have EVER matched each prediction

    Matching rule per frame:
      - match if IoU >= iou_thr
        OR IoP >= ios_thr
      - if use_void_tolerant=True, prediction pixels falling into GT void
        are ignored in IoU / IoP area accounting

    Returns:
      ic_p: prediction-axis IC, shape [num_pred_ids]
      ic_g: GT-axis IC, shape [num_gt_ids]

    Definitions:
      IC_G(g) = 1 / (# distinct predictions ever matched to g)
      IC_P(p) = 1 / (# distinct GTs ever matched to p)

    Unmatched ids contribute 0.
    """
    num_p = len(p_ids)
    num_g = len(g_ids)

    # Track ever-matched relations over the whole sequence
    matched_preds_for_gt = {gid: set() for gid in g_ids}
    matched_gts_for_pred = {pid: set() for pid in p_ids}

    for ps, gs in zip(seg_list, gt_list):
        p_arr = _to_numpy(ps).astype(np.int64)
        g_arr = _to_numpy(gs).astype(np.int64)

        p_masks_raw, p_areas_raw = _get_frame_masks(p_arr, ignore_label)
        g_masks, g_areas = _get_frame_masks(g_arr, ignore_label)

        valid_gt_mask = (g_arr != ignore_label)

        # Void-tolerant prediction masks/areas if requested
        if use_void_tolerant:
            p_masks = {}
            p_areas = {}
            for pid, pmask in p_masks_raw.items():
                pmask_valid = np.logical_and(pmask, valid_gt_mask)
                area_valid = int(pmask_valid.sum())
                if area_valid > 0:
                    p_masks[pid] = pmask_valid
                    p_areas[pid] = area_valid
        else:
            p_masks = p_masks_raw
            p_areas = p_areas_raw

        # Frame-level matching, but relations are accumulated globally
        for pid, pmask in p_masks.items():
            p_area = p_areas[pid]
            if p_area == 0:
                continue

            for gid, gmask in g_masks.items():
                g_area = g_areas[gid]
                inter = int(np.logical_and(pmask, gmask).sum())
                if inter == 0:
                    continue

                union = p_area + g_area - inter
                if union <= 0:
                    continue

                iou = inter / union
                iop = inter / max(p_area, 1)

                # Match if either IoU or prediction-side support is sufficient.
                if iou >= iou_thr or (iou < iou_thr and iop >= ios_thr):
                    matched_preds_for_gt[gid].add(pid)
                    matched_gts_for_pred[pid].add(gid)

    # Prediction-axis identity concentration.
    ic_p = np.zeros(num_p, dtype=np.float64)

    for pid, idx_p in p_map.items():
        n = len(matched_gts_for_pred[pid])
        ic_p[idx_p] = (1.0 / n) if n > 0 else 0.0

    # GT-axis identity concentration.
    ic_g = np.zeros(num_g, dtype=np.float64)

    for gid, idx_g in g_map.items():
        matched_pids = matched_preds_for_gt[gid]
        n = len(matched_pids)

        if n == 0:
            ic_g[idx_g] = 0.0
            continue

        ic_g[idx_g] = 1.0 / n

    return ic_p, ic_g

def compute_sequence_stq_vpq(
    preds,
    gts,
    ignore_label=0,
    vpq_iou_threshold=0.5,
    k_values=(1, 5, 10, 15),
    ios_threshold=0.5,
    area_penalty_power=1.0,
    sever_ratio_threshold=0.5,
    use_soft_stability=True,
    use_dominant_fragment=True,
):
    """
    Computes class-agnostic STQ and sliding-window VPQ_k utilizing n-to-1 part-whole matching.

    Key behavior:
      - void-tolerant: prediction spill into GT void is ignored
      - GT-referenced one-sided assignment by IoS = inter / pred_area
      - sever points are detected from area-weighted persistent-support ratio
      - support is segmented into fragments on the non-empty GT-present support chain
      - dominant fragment is selected by total GT-overlap mass
      - only the dominant fragment contributes to GT TP / IoU / matched_preds
      - all fragment predictions of any GT are protected from ordinary FP
      - fragment-level FP units are added as (#fragments - 1), regardless of GT match success
      - ordinary unmatched predictions are still counted as usual
    """
    from collections import defaultdict
    import numpy as np

    # Frame and global bookkeeping with void-tolerant prediction areas.
    frame_gt_areas = []
    frame_pred_areas = []
    frame_intersections = []

    total_fg_inter = 0
    total_fg_union = 0
    global_gt_areas = defaultdict(int)
    global_pred_areas = defaultdict(int)
    global_intersections = defaultdict(lambda: defaultdict(int))

    for p, g in zip(preds, gts):
        p = np.asarray(p)
        g = np.asarray(g)

        p_flat, g_flat = p.ravel(), g.ravel()

        g_valid_mask = (g_flat != ignore_label)
        p_valid_mask = (p_flat != ignore_label) & g_valid_mask

        total_fg_inter += p_valid_mask.sum()
        total_fg_union += g_valid_mask.sum()

        p_ids, p_counts = np.unique(p_flat[p_valid_mask], return_counts=True)
        cur_p_areas = dict(zip(p_ids.tolist(), p_counts.tolist()))
        frame_pred_areas.append(cur_p_areas)
        for pid, c in cur_p_areas.items():
            global_pred_areas[pid] += c

        g_ids, g_counts = np.unique(g_flat[g_valid_mask], return_counts=True)
        cur_g_areas = dict(zip(g_ids.tolist(), g_counts.tolist()))
        frame_gt_areas.append(cur_g_areas)
        for gid, c in cur_g_areas.items():
            global_gt_areas[gid] += c

        cur_inter = defaultdict(int)
        if p_valid_mask.any():
            for gid, pid in zip(g_flat[p_valid_mask], p_flat[p_valid_mask]):
                gid = int(gid)
                pid = int(pid)
                cur_inter[(gid, pid)] += 1
                global_intersections[gid][pid] += 1
        frame_intersections.append(cur_inter)

    # Global prediction-to-GT assignment by IoS.
    gq = total_fg_inter / (total_fg_union + 1e-15)

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
            global_pred_to_gt[pid] = best_gt

    global_gt_virtual_preds = defaultdict(list)
    for pid, gid in global_pred_to_gt.items():
        global_gt_virtual_preds[gid].append(pid)

    # Helper functions for support-chain scoring.
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
        """
        Returns non-empty GT-present support chain:
          chain = [(frame_idx, active_ids, support_mass), ...]
        where support_mass is total GT-overlap mass carried by active_ids at that frame.
        Empty GT-present frames are skipped.
        """
        chain = []
        for f in frame_range:
            if gid not in gt_area_lookup[f]:
                continue
            active = {
                p for p in assigned_preds
                if inter_lookup[f].get((gid, p), 0) > 0
            }
            if not active:
                continue
            support_mass = sum(inter_lookup[f].get((gid, p), 0) for p in active)
            chain.append((f, active, support_mass))
        return chain

    def sever_ratios_from_chain(gid, chain, inter_lookup):
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

    def dominant_fragment_from_chain(gid, chain, inter_lookup):
        """
        Split support chain by sever points and select dominant fragment by total GT-overlap mass.

        Returns:
          dominant_frames: list[int]
          dominant_pred_ids: set[int]
          fragment_info: list[dict]
          sever_ratios: list[float]
          sever_flags: list[bool]
        """
        if not use_dominant_fragment or len(chain) == 0:
            return [], set(), [], [], []

        sever_ratios, sever_flags = sever_ratios_from_chain(gid, chain, inter_lookup)

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
                "start_idx": a,
                "end_idx": b,
                "frames": frames,
                "pred_ids": pred_ids,
                "mass": mass,
                "length": len(frames),
            })

        fragment_info.sort(key=lambda x: (x["mass"], x["length"]), reverse=True)
        dom = fragment_info[0]

        return dom["frames"], dom["pred_ids"], fragment_info, sever_ratios, sever_flags

    def score_gt_with_fragment(
        gid,
        g_area,
        selected_preds,
        selected_frames,
        pred_area_lookup,
        inter_lookup,
        area_lookup_for_gt,
    ):
        if not selected_preds or not selected_frames:
            return {
                "virtual_tpa": 0,
                "virtual_p_area": 0,
                "union": g_area,
                "raw_iou": 0.0,
                "soft_stability": 0.0,
                "area_penalty": 0.0,
                "consistency_penalty": 0.0,
                "final_iou": 0.0,
                "matched_preds_for_gt": set(),
            }

        virtual_tpa = 0
        for f in selected_frames:
            virtual_tpa += sum(inter_lookup[f].get((gid, p), 0) for p in selected_preds)

        virtual_p_area = 0
        for f in selected_frames:
            virtual_p_area += sum(
                pred_area_lookup[f].get(p, 0) for p in selected_preds if p in pred_area_lookup[f]
            )

        union = g_area + virtual_p_area - virtual_tpa
        raw_iou = virtual_tpa / max(union, 1)

        if use_soft_stability:
            active_sets = []
            for f in selected_frames:
                if gid not in area_lookup_for_gt[f]:
                    continue
                active = {
                    p for p in selected_preds
                    if inter_lookup[f].get((gid, p), 0) > 0
                }
                if active:
                    active_sets.append(active)
            soft_stability = mean_consecutive_jaccard(active_sets)
        else:
            soft_stability = 1.0

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
            "virtual_tpa": virtual_tpa,
            "virtual_p_area": virtual_p_area,
            "union": union,
            "raw_iou": raw_iou,
            "soft_stability": soft_stability,
            "area_penalty": area_penalty,
            "consistency_penalty": consistency_penalty,
            "final_iou": final_iou,
            "matched_preds_for_gt": matched_preds_for_gt,
        }

    # STQ / AQ using the dominant fragment only.
    aq_per_gt = {}
    for gid, g_area in global_gt_areas.items():
        assigned_preds = global_gt_virtual_preds.get(gid, [])
        if not assigned_preds:
            aq_per_gt[gid] = 0.0
            continue

        chain = build_support_chain(
            gid, assigned_preds, range(len(preds)),
            frame_intersections, frame_gt_areas
        )
        dom_frames, dom_preds, _, _, _ = dominant_fragment_from_chain(
            gid, chain, frame_intersections
        )

        gt_score = score_gt_with_fragment(
            gid,
            g_area,
            dom_preds,
            dom_frames,
            frame_pred_areas,
            frame_intersections,
            frame_gt_areas,
        )

        inner_sum = gt_score["virtual_tpa"] * gt_score["raw_iou"] * gt_score["consistency_penalty"]
        aq_per_gt[gid] = inner_sum

    aq = (
        sum(val / global_gt_areas[gid] for gid, val in aq_per_gt.items()) / len(global_gt_areas)
        if global_gt_areas else 0.0
    )
    stq = np.sqrt(aq * gq)

    # Sliding-window VPQ_k using the dominant fragment only.
    num_frames = len(preds)
    vpq_scores = {}

    for k in k_values:
        actual_k = 1 if k == 0 else (num_frames if k == float("inf") else min(k, num_frames))
        window_pqs = []

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

            # window-local assignment by IoS
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

            fragment_fp = 0
            protected_assigned_preds = set()  # all preds belonging to any GT fragment in this window

            window_range = range(t, t + actual_k)

            # First pass: build fragment protection for all GTs with fragment support
            per_gt_fragment_cache = {}
            for gid, g_area in win_gt_areas.items():
                assigned_preds = win_gt_virtual_preds.get(gid, [])
                if not assigned_preds:
                    continue

                chain = build_support_chain(
                    gid, assigned_preds, window_range,
                    frame_intersections, frame_gt_areas
                )
                if not chain:
                    continue

                dom_frames, dom_preds, fragment_info, _, _ = dominant_fragment_from_chain(
                    gid, chain, frame_intersections
                )
                per_gt_fragment_cache[gid] = (dom_frames, dom_preds, fragment_info)

                # Protect all fragment predictions regardless of final GT match success
                if use_dominant_fragment:
                    for frag in fragment_info:
                        protected_assigned_preds |= frag["pred_ids"]

                    # Fragment-level FP applies regardless of match success
                    fragment_fp += max(len(fragment_info) - 1, 0)

            # Second pass: compute GT matching using dominant fragment only
            for gid, g_area in win_gt_areas.items():
                if gid not in per_gt_fragment_cache:
                    continue

                dom_frames, dom_preds, fragment_info = per_gt_fragment_cache[gid]

                gt_score = score_gt_with_fragment(
                    gid,
                    g_area,
                    dom_preds,
                    dom_frames,
                    frame_pred_areas,
                    frame_intersections,
                    frame_gt_areas,
                )

                iou = gt_score["final_iou"]

                if iou > vpq_iou_threshold:
                    tp += 1
                    iou_sum += iou
                    matched_gts.add(gid)
                    matched_preds |= gt_score["matched_preds_for_gt"]

            ordinary_fp = 0
            for pid in win_pred_areas.keys():
                # rescued by dominant matched support
                if pid in matched_preds:
                    continue

                # belongs to some GT fragment support: do not count as ordinary FP
                if pid in protected_assigned_preds:
                    continue

                inter_with_valid_gts = sum(
                    win_inter.get((gid, pid), 0) for gid in win_gt_areas.keys()
                )
                void_area = win_pred_areas[pid] - inter_with_valid_gts

                if void_area / max(win_pred_areas[pid], 1) > 0.5:
                    pass
                else:
                    ordinary_fp += 1

            fp = ordinary_fp + fragment_fp
            fn = len(win_gt_areas) - len(matched_gts)

            denom = tp + 0.5 * fp + 0.5 * fn
            pq = iou_sum / denom if denom > 0 else 0.0
            window_pqs.append(pq)

        key_name = "VPQ_inf" if k == float("inf") else f"VPQ_{k}"
        vpq_scores[key_name] = sum(window_pqs) / len(window_pqs) if window_pqs else 0.0

    standard_ks = [v for k, v in vpq_scores.items() if k != "VPQ_inf"]
    vpq_avg = sum(standard_ks) / len(standard_ks) if standard_ks else 0.0

    results = {"STQ": stq, "AQ": aq, "GQ": gq, "VPQ_avg": vpq_avg}
    results.update(vpq_scores)
    return results

def _compute_pattern_coverage(pred_rows_j, 
                              pred_frames_for_gt_j, 
                              frames_j, 
                              max_pattern_size):
    T_j = len(frames_j)
    scores_k = {k: 0.0 for k in range(1, max_pattern_size + 1)}
    if T_j == 0 or not pred_rows_j: return scores_k, []
    
    frames_j_set = set(frames_j)
    frame_sets = [pred_frames_for_gt_j[pid] & frames_j_set 
                  for pid in pred_rows_j]
    best_overall_rows = []
    best_overall_supp = 0

    for k in range(1, min(max_pattern_size, len(pred_rows_j)) + 1):
        best_support, best_combo = 0, []
        for idx_combo in combinations(range(len(pred_rows_j)), k):
            inter = frame_sets[idx_combo[0]].copy()
            for li in idx_combo[1:]:
                inter &= frame_sets[li]
                if not inter: break
            if len(inter) > best_support:
                best_support, best_combo = len(inter), list(idx_combo)
        
        scores_k[k] = best_support / T_j
        if best_support > best_overall_supp:
            best_overall_supp = best_support
            best_overall_rows = [pred_rows_j[li] for li in best_combo]
            
    return scores_k, best_overall_rows

def _compute_temporal_stability(gt_ids, 
                                pred_ids, 
                                frames_with_gt, 
                                active_preds_per_gt, 
                                pred_frames_per_gt, 
                                cooccur_sim_thr, 
                                max_pattern_size):
    num_gt = len(gt_ids)
    stability_scores = np.zeros(num_gt)
    cluster_scores = np.zeros(num_gt)
    patt_curves = []

    for j in range(num_gt):
        T_j = len(frames_with_gt[j])
        pred_rows_j, occurs, cooccur, adjacent = \
            _build_cooccurrence_for_gt(pred_frames_per_gt[j], 
                                       active_preds_per_gt[j])
            
        if T_j == 0 or not pred_rows_j:
            patt_curves.append({k: 0.0 for k in range(1, max_pattern_size+1)})
            continue

        # Cluster
        sim = _build_similarity_graph(occurs, cooccur, adjacent, cooccur_sim_thr)
        clusters = _connected_components_from_similarity(sim, cooccur_sim_thr)
        max_cov = 0
        for comp in clusters:
            union = set()
            for li in comp: union |= pred_frames_per_gt[j][pred_rows_j[li]]
            max_cov = max(max_cov, len(union))
        beta_c = max_cov / T_j
        
        # Pattern
        sk, _ = _compute_pattern_coverage(pred_rows_j, 
                                          pred_frames_per_gt[j], 
                                          frames_with_gt[j], 
                                          max_pattern_size)
        patt_curves.append(sk)
        beta_p = max(sk.values())
        
        # Harmonic TS (Comb)
        cluster_scores[j] = beta_c
        
        if (beta_c + beta_p) > 0:
            stability_scores[j] = (2 * beta_c * beta_p) / (beta_c + beta_p)
        else:
            stability_scores[j] = 0.0

    total_p = {k: float(np.mean([c[k] for c in patt_curves])) \
               for k in range(1, max_pattern_size+1)}
               
    return (float(np.mean(stability_scores)), 
            float(np.mean(cluster_scores)), 
            total_p, 
            stability_scores, 
            cluster_scores, 
            patt_curves)

def _build_cooccurrence_for_gt(pf_gt, ap_gt):
    p_rows = sorted(pf_gt.keys())
    if not p_rows: return [], None, None, None  
    g2l = {p: i for i, p in enumerate(p_rows)}
    occurs = np.zeros(len(p_rows))
    for p, f in pf_gt.items(): occurs[g2l[p]] = len(f)
    
    co = np.zeros((len(p_rows), len(p_rows)))
    adjacent = np.zeros((len(p_rows), len(p_rows)), dtype=bool) 
    
    time_steps = sorted(ap_gt.keys())
    for idx, t in enumerate(time_steps):
        current_acts = ap_gt[t]
        l_idxs = [g2l[p] for p in current_acts if p in g2l]
        
        # Standard co-occurrence.
        for a, b in combinations(l_idxs, 2): 
            co[a, b] = co[b, a] = co[a, b] + 1
            
        # Adjacent-frame handoff between prediction IDs.
        if idx < len(time_steps) - 1 and time_steps[idx + 1] == t + 1:
            next_acts = ap_gt[t + 1]
            for p_curr in current_acts:
                for p_next in next_acts:
                    if p_curr != p_next and p_curr in g2l and p_next in g2l:
                        a, b = g2l[p_curr], g2l[p_next]
                        adjacent[a, b] = adjacent[b, a] = True
                        
    return p_rows, occurs, co, adjacent

def _build_similarity_graph(occ, co, adjacent, *args):
    """
    Builds an unweighted adjacency graph.
    If two masks overlap in time AT ALL (co > 0), or handshake end-to-end,
    they are connected in the cluster graph. 
    """
    sim = np.zeros((len(occ), len(occ)))
    for a, b in combinations(range(len(occ)), 2):
        if co[a, b] > 0 or adjacent[a, b]:
            sim[a, b] = sim[b, a] = 1.0  
    return sim

def _connected_components_from_similarity(sim, thr):
    visited, comps = [False] * len(sim), []
    for i in range(len(sim)):
        if not visited[i]:
            stack, c = [i], []
            visited[i] = True
            while stack:
                u = stack.pop(); c.append(u)
                for v in np.where(sim[u] >= thr)[0]:
                    if not visited[v]: visited[v] = True; stack.append(v)
            comps.append(c)
    return comps

def evaluate_vos_consistency(seg_list, 
                             gt_list, 
                             ignore_label=0, 
                             iou_thr=0.5, 
                             ios_thr=0.7, 
                             cooccur_sim_thr=0.5, 
                             max_pattern_size=2):
    
    p_ids, p_map = _collect_global_ids(seg_list, ignore_label)
    g_ids, g_map = _collect_global_ids(gt_list, ignore_label)
    
    fwg = [set() for _ in range(len(g_ids))]
    apg = [defaultdict(set) for _ in range(len(g_ids))]
    pfg = [defaultdict(set) for _ in range(len(g_ids))]
    
    g_area_sums = np.zeros(len(g_ids))
    g_area_counts = np.zeros(len(g_ids))
    p_area_sums = np.zeros(len(p_ids))
    p_area_counts = np.zeros(len(p_ids))

    # Purity and discovery-tax accumulators.
    global_inters = np.zeros((len(p_ids), len(g_ids)))
    spatial_purity_sums = np.zeros(len(p_ids))
    pred_frame_counts = np.zeros(len(p_ids))
    frame_discovery_factors = []

    for t, (ps, gs) in enumerate(zip(seg_list, gt_list)):
        p_arr = _to_numpy(ps).astype(np.int64)
        g_arr = _to_numpy(gs).astype(np.int64)

        # GT masks and areas stay unchanged.
        g_masks, g_areas = _get_frame_masks(g_arr, ignore_label)

        # Ignore prediction spill into GT void regions.
        valid_gt_mask = (g_arr != ignore_label)
        p_masks_raw, _ = _get_frame_masks(p_arr, ignore_label)

        p_masks = {}
        p_areas = {}
        for pid, pmask in p_masks_raw.items():
            pmask_valid = np.logical_and(pmask, valid_gt_mask)
            area_valid = int(pmask_valid.sum())
            if area_valid > 0:
                p_masks[pid] = pmask_valid
                p_areas[pid] = area_valid
        
        for gid, area in g_areas.items():
            idx = g_map[gid]
            fwg[idx].add(t)
            g_area_sums[idx] += area
            g_area_counts[idx] += 1
            
        for pid, area in p_areas.items():
            idx = p_map[pid]
            p_area_sums[idx] += area
            p_area_counts[idx] += 1
            
        matches, f_inters = _compute_frame_matches(
            p_masks,
            p_areas,
            g_masks,
            g_areas,
            iou_thr,
            ios_thr,
            valid_gt_mask=None,
        )
        
        # Per-frame discovery factor.
        matched_p_ids_this_frame = set()
        for gid, pids in matches.items():
            idx_g = g_map[gid]
            for pid in pids:
                idx_p = p_map[pid]
                matched_p_ids_this_frame.add(pid)
                apg[idx_g][t].add(idx_p)
                pfg[idx_g][idx_p].add(t)

        n_gt_t = max(len(g_areas), 1)
        f_discovery_tax = min(len(matched_p_ids_this_frame) / n_gt_t, 1.0)
        frame_discovery_factors.append(f_discovery_tax)

        # Per-prediction spatial purity.
        for pid, g_inter_dict in f_inters.items():
            idx_p = p_map[pid]
            max_i = max(g_inter_dict.values())
            p_area_t = p_areas[pid]
            
            # Void-tolerant local precision.
            local_precision = (max_i / p_area_t) if p_area_t > 0 else 0.0
            
            spatial_purity_sums[idx_p] += (local_precision * f_discovery_tax)
            pred_frame_counts[idx_p] += 1
            
            for gid, inter in g_inter_dict.items():
                global_inters[idx_p, g_map[gid]] += inter

    # Aggregate identity-persistence terms.
    avg_discovery_tax = np.mean(frame_discovery_factors) if frame_discovery_factors else 0.0
    
    # Prediction axis: global precision scaled by scene-level discovery.
    p_scores_global = np.zeros(len(p_ids))
    valid_p = p_area_sums > 0
    if np.any(valid_p) and global_inters.shape[1] > 0:
        p_scores_global[valid_p] = (
            global_inters[valid_p].max(axis=1) / p_area_sums[valid_p]
        ) * avg_discovery_tax

    # GT axis: GT recall scaled by scene-level discovery.
    g_scores_global = np.zeros(len(g_ids))
    valid_g = g_area_sums > 0
    if np.any(valid_g) and global_inters.shape[0] > 0:
        g_scores_global[valid_g] = (
            global_inters[:, valid_g].max(axis=0) / g_area_sums[valid_g]
        ) * avg_discovery_tax

    # Spatial purity uses void-tolerant prediction areas.
    p_scores_spatial = np.zeros(len(p_ids))
    valid_f = pred_frame_counts > 0
    if np.any(valid_f):
        p_scores_spatial[valid_f] = spatial_purity_sums[valid_f] / pred_frame_counts[valid_f]

    # Means over all IDs.
    p_avg = float(np.mean(p_scores_global)) if len(p_scores_global) > 0 else 0.0
    g_avg = float(np.mean(g_scores_global)) if len(g_scores_global) > 0 else 0.0
    c_ip = (2 * p_avg * g_avg) / (p_avg + g_avg) if (p_avg + g_avg) > 0 else 0.0
    
    p_spat_avg = float(np.mean(p_scores_spatial)) if len(p_scores_spatial) > 0 else 0.0
    temporal_bleeds = np.maximum(0, p_scores_spatial - p_scores_global)
    t_bleed_avg = float(np.mean(temporal_bleeds)) if len(temporal_bleeds) > 0 else 0.0

    # Structural identity concentration.
    ic_p_arr, ic_g_arr = _compute_identity_concentration(
        seg_list,
        gt_list,
        p_ids,
        g_ids,
        p_map,
        g_map,
        ignore_label=ignore_label,
        iou_thr=iou_thr,
        ios_thr=ios_thr,
        use_void_tolerant=True,
    )

    ic_p_avg = float(np.mean(ic_p_arr)) if len(ic_p_arr) > 0 else 0.0
    ic_g_avg = float(np.mean(ic_g_arr)) if len(ic_g_arr) > 0 else 0.0

    # Temporal stability.
    c_ts, c_clus, c_curve, ts_raw, clus_raw, curve_raw = \
        _compute_temporal_stability(g_ids, p_ids, fwg, apg, pfg, 
                                    cooccur_sim_thr, max_pattern_size)

    # Baseline metrics.
    np_preds = [_to_numpy(p) for p in seg_list]
    np_gts = [_to_numpy(g) for g in gt_list]
    baseline_metrics = compute_sequence_stq_vpq(
        np_preds, np_gts, ignore_label=ignore_label,
        vpq_iou_threshold=0.5, k_values=(0, 5, 15, 25, 35, float('inf')),
    )

    results = {
        "baselines": baseline_metrics,
        "identity_persistence": {
            "combined": c_ip, 
            "prediction_axis": p_avg, 
            "gt_axis": g_avg,
            "ic_p": ic_p_avg,
            "ic_g": ic_g_avg,
            "pred_spatial": p_spat_avg,
            "temporal_bleed": t_bleed_avg,
            "discovery_tax": avg_discovery_tax
        },
        "temporal_stability": { "combined": c_ts, "cluster": c_clus, "pattern_curve": c_curve },
        "size_breakdown": {}
    }
    
    h, w = _to_numpy(gt_list[0]).shape[:2]

    g_sz = [
        _get_size_label(s / c, h, w) if c > 0 else 'S'
        for s, c in zip(g_area_sums, g_area_counts)
    ]
    p_sz = [
        _get_size_label(s / c, h, w) if c > 0 else 'S'
        for s, c in zip(p_area_sums, p_area_counts)
    ]

    for sz in ['S', 'M', 'L']:
        m_p = np.array(p_sz) == sz
        m_g = np.array(g_sz) == sz
        
        ip_p = np.mean(p_scores_global[m_p]) if any(m_p) else 0.0
        ip_g = np.mean(g_scores_global[m_g]) if any(m_g) else 0.0
        
        ic_p = np.mean(ic_p_arr[m_p]) if any(m_p) else 0.0
        ic_g = np.mean(ic_g_arr[m_g]) if any(m_g) else 0.0
        
        ip_spat = np.mean(p_scores_spatial[m_p]) if any(m_p) else 0.0
        t_bleed = np.mean(temporal_bleeds[m_p]) if any(m_p) else 0.0
        
        ts_v = np.mean(ts_raw[m_g]) if any(m_g) else 0.0
        cl_v = np.mean(clus_raw[m_g]) if any(m_g) else 0.0
        cv_v = {k: np.mean([c[k] for i, c in enumerate(curve_raw) if m_g[i]]) \
                if any(m_g) else 0.0 for k in range(1, max_pattern_size+1)}
                
        results["size_breakdown"][sz] = {
            "ip_comb": (2*ip_p*ip_g)/(ip_p+ip_g) if (ip_p+ip_g)>0 else 0.0,
            "ip_p": ip_p, 
            "ip_g": ip_g, 
            "ic_p": ic_p,
            "ic_g": ic_g,
            "ip_spat": ip_spat,
            "t_bleed": t_bleed,
            "ts_c": ts_v, 
            "cl_v": cl_v, 
            "curve": cv_v, 
            "count": int(sum(m_g))
        }
    return results
