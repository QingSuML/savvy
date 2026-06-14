import csv
import gc
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .components import (
    ObjectLocationFeature,
    HandshakeBuffer,
    SmoothedMaskFeedbackGridManager,
    _bootstrap_initial_frame,
    _collect_obj_ptrs_and_masks,
    _find_contained_parts,
    _flush_new_obj_temp_outputs,
    _merge_transient_into_established,
    _prune_old_non_cond_outputs,
    _strip_pred_masks_from_old_frames,
    _suppress_tqdm,
    apply_seniority_suppression_to_frame,
    backfill_transient_gaps,
    hierarchical_mask_merge,
    load_external_masks_per_frame,
    new_mask_fn,
)

__all__ = ["Savvy", "load_external_masks_per_frame"]


class Savvy:
    def __init__(
        self,
        predictor,
        inference_state,
        segmenter=None,
        buffer_size=30,
        iou_threshold=0.5,
        survival_threshold=0.1,
        margin=0.05,
        memory_strength=0.1,
        max_bank_size=None,
        max_active_objects=300,
        min_area_to_keep=1500,
        prune_memory_every_n_frames=10,
        sam_num_points_per_side=32,
        segmenter_stride=3,
        adapt_num_points=False,
        exploratory=False,
        behavior_log_path=None,
        handshake_min_agreement_hits=5,
        handshake_min_visible_frames_for_promotion=3,
        handshake_min_max_area_for_promotion=-1,
        handshake_min_max_area_ratio_for_promotion=-1.0,
    ):
        # SAM2 state.
        self.predictor = predictor
        self.inference_state = inference_state

        # Optional SAM1 mask generator.
        self.segmenter = segmenter
        self.sam_num_points_per_side = sam_num_points_per_side
        self.segmenter_stride = segmenter_stride

        # Tracking configuration.
        self.buffer_size = buffer_size
        self.iou_threshold = iou_threshold
        self.survival_threshold = survival_threshold
        self.margin = margin
        self.memory_strength = memory_strength
        self.max_bank_size = max_bank_size
        self.prune_memory_every_n_frames = prune_memory_every_n_frames
        self.adapt_num_points = adapt_num_points
        self.max_active_objects = max_active_objects
        self.min_area_to_keep = min_area_to_keep

        self.handshake_min_agreement_hits = handshake_min_agreement_hits
        self.handshake_min_visible_frames_for_promotion = (
            handshake_min_visible_frames_for_promotion
        )
        self.handshake_min_max_area_for_promotion = handshake_min_max_area_for_promotion
        self.handshake_min_max_area_ratio_for_promotion = (
            handshake_min_max_area_ratio_for_promotion
        )
        self.exploratory=exploratory
        self.behavior_log_path = behavior_log_path
        self._behavior_log_file = None
        self._behavior_log_writer = None
        self._behavior_frame_count = 0
        self._behavior_total_seconds = 0.0
        self._behavior_frame_stats = {}
        self._image_hw = (
            inference_state["video_height"],
            inference_state["video_width"],
        )
        self._num_frames = inference_state["num_frames"]

        # Initialize mutable tracking state.
        self.reset_state()

        # Per-object and per-frame tracking state.
        self.all_results: Dict[int, Dict[int, np.ndarray]] = {}
        self.transient_mask_buffer: Dict[int, Dict[int, np.ndarray]] = {}
        self.pending_absorptions: List[Tuple[int, int]] = []
        self.established_miss_streak: Dict[int, int] = {}
        self.continuous_presence_streak: Dict[int, int] = {}
        self.next_obj_id: int = 0

        # Helper components.
        self.obj_location_feature = ObjectLocationFeature(
            box_scale=1.5,
            ema_alpha=0.9,
            max_bank_size=max_bank_size,
        )
        self.buffer = HandshakeBuffer(
            buffer_size=self.buffer_size,
            iou_threshold=self.iou_threshold,
            min_agreement_hits=self.handshake_min_agreement_hits,
            min_visible_frames_for_promotion=self.handshake_min_visible_frames_for_promotion,
            min_max_area_for_promotion=self.handshake_min_max_area_for_promotion,
            min_max_area_ratio_for_promotion=self.handshake_min_max_area_ratio_for_promotion,
            image_hw=self._image_hw,
        )

        # SAM2 runtime settings.
        self.predictor.max_cond_frames_in_attn = 4

    def _open_behavior_log(self):
        if self.behavior_log_path is None:
            return

        log_dir = os.path.dirname(self.behavior_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._behavior_log_file = open(self.behavior_log_path, "w", newline="")
        fieldnames = [
            "frame_idx",
            "frame_ordinal",
            "frame_tracking_seconds",
            "instant_tracking_fps",
            "cumulative_tracking_fps",
            "segmenter_invoked",
            "segmenter_candidate_mask_count",
            "newly_discovered_mask_count",
            "object_set_size",
            "active_object_count",
            "established_object_count",
            "transient_object_count",
            "propagated_mask_count",
            "total_mask_count",
            "cuda_device_used_mb",
            "cuda_memory_reserved_mb",
            "cuda_memory_allocated_mb",
        ]
        self._behavior_log_writer = csv.DictWriter(
            self._behavior_log_file, fieldnames=fieldnames
        )
        self._behavior_log_writer.writeheader()
        self._behavior_frame_count = 0
        self._behavior_total_seconds = 0.0

    def _close_behavior_log(self):
        if self._behavior_log_file is not None:
            self._behavior_log_file.close()
        self._behavior_log_file = None
        self._behavior_log_writer = None

    def _cuda_memory_stats_mb(self):
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0

        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        used_mb = (total_bytes - free_bytes) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
        allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        return used_mb, reserved_mb, allocated_mb

    def _count_segmenter_candidates(self, external_masks):
        if external_masks is None:
            return 0
        if torch.is_tensor(external_masks):
            arr = external_masks.detach().cpu().numpy()
        else:
            arr = np.asarray(external_masks)
        if arr.ndim == 3:
            return int(arr.shape[0])
        labels = np.unique(arr)
        return int(np.count_nonzero(labels))

    def _start_behavior_frame(self):
        self._behavior_frame_stats = {
            "segmenter_invoked": 0,
            "segmenter_candidate_mask_count": 0,
            "newly_discovered_mask_count": 0,
        }

    def _write_behavior_frame(self, frame_idx, started_at, propagated_mask_count):
        if self._behavior_log_writer is None:
            return

        elapsed = time.perf_counter() - started_at
        self._behavior_frame_count += 1
        self._behavior_total_seconds += elapsed

        established_visible = len(self.all_results.get(frame_idx, {}))
        transient_visible = sum(
            1
            for frame_masks in self.transient_mask_buffer.values()
            if frame_idx in frame_masks and np.squeeze(frame_masks[frame_idx]).any()
        )
        active_object_count = len(self.inference_state.get("obj_id_to_idx", {}))
        cuda_used, cuda_reserved, cuda_allocated = self._cuda_memory_stats_mb()

        self._behavior_log_writer.writerow({
            "frame_idx": frame_idx,
            "frame_ordinal": self._behavior_frame_count - 1,
            "frame_tracking_seconds": elapsed,
            "instant_tracking_fps": 1.0 / elapsed if elapsed > 0 else 0.0,
            "cumulative_tracking_fps": (
                self._behavior_frame_count / self._behavior_total_seconds
                if self._behavior_total_seconds > 0
                else 0.0
            ),
            "segmenter_invoked": self._behavior_frame_stats["segmenter_invoked"],
            "segmenter_candidate_mask_count": self._behavior_frame_stats[
                "segmenter_candidate_mask_count"
            ],
            "newly_discovered_mask_count": self._behavior_frame_stats[
                "newly_discovered_mask_count"
            ],
            "object_set_size": active_object_count,
            "active_object_count": active_object_count,
            "established_object_count": len(self.established_miss_streak),
            "transient_object_count": len(self.buffer.transients),
            "propagated_mask_count": propagated_mask_count,
            "total_mask_count": established_visible + transient_visible,
            "cuda_device_used_mb": cuda_used,
            "cuda_memory_reserved_mb": cuda_reserved,
            "cuda_memory_allocated_mb": cuda_allocated,
        })
        self._behavior_log_file.flush()

    def reset_state(self):
        """
        Manually clears all tracking dictionaries and re-initializes 
        sub-components to release PyTorch tensors and free memory.
        """
        self.all_results: Dict[int, Dict[int, np.ndarray]] = {}
        self.transient_mask_buffer: Dict[int, Dict[int, np.ndarray]] = {}
        self.pending_absorptions: List[Tuple[int, int]] = []
        self.established_miss_streak: Dict[int, int] = {}
        self.continuous_presence_streak: Dict[int, int] = {}
        self.next_obj_id: int = 0
        self.max_historical_area: Dict[int, int] = {}
        self.recent_evictions = {}

        image_hw = getattr(self, "_image_hw", (
            self.inference_state["video_height"],
            self.inference_state["video_width"],
        ))

        self.grid_manager = SmoothedMaskFeedbackGridManager(
            base_points=self.sam_num_points_per_side,
            min_points=max(8, self.sam_num_points_per_side // 4),
            max_points=self.sam_num_points_per_side
        )

        self.obj_location_feature = ObjectLocationFeature(
            box_scale=1.5,
            ema_alpha=0.9,
            max_bank_size=self.max_bank_size,
        )
        self.buffer = HandshakeBuffer(
            buffer_size=self.buffer_size,
            iou_threshold=self.iou_threshold,
            min_agreement_hits=self.handshake_min_agreement_hits,
            min_visible_frames_for_promotion=self.handshake_min_visible_frames_for_promotion,
            min_max_area_for_promotion=self.handshake_min_max_area_for_promotion,
            min_max_area_ratio_for_promotion=self.handshake_min_max_area_ratio_for_promotion,
            image_hw=image_hw,
        )
        gc.collect() 
        torch.cuda.empty_cache()

    def _get_current_transient_masks(self, frame_idx: int):
        """
        Return all non-empty transient masks at the given frame.
        """
        masks = []
        for obj_id, frame_masks in self.transient_mask_buffer.items():
            mask = frame_masks.get(frame_idx)
            if mask is not None and np.squeeze(mask).any():
                masks.append(mask)
        return masks

    @torch.inference_mode()
    def _query_segmenter(self, frame_idx: int, num_points_per_side=None) -> Optional[torch.Tensor]:
        device = self.inference_state["device"]
        H, W = self._image_hw

        # Convert the normalized SAM2 tensor back to uint8 HWC for SAM1.
        image = self.inference_state["images"][frame_idx]  # [3, 1024, 1024]
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        
        image_np = (
            (image * std + mean)
            .clamp(0, 1)
            .permute(1, 2, 0)
            .to(torch.float32)  
            .cpu()
            .numpy() * 255
        ).astype(np.uint8)  
        
        # Downsample before SAM1 mask generation for speed.
        image_np = cv2.resize(image_np, (512, 512), interpolation=cv2.INTER_LINEAR)

        # Choose the SAM1 prompt grid density.
        n = num_points_per_side or self.sam_num_points_per_side
        if self.adapt_num_points:
            n = self.grid_manager.current_grid

        # Build a vacancy mask so SAM1 is queried mainly in uncovered regions.
        forward_mask = None

        established_masks = [
            mask
            for mask in self.all_results.get(frame_idx, {}).values()
            if mask is not None and np.squeeze(mask).any()
        ]
        transient_masks = self._get_current_transient_masks(frame_idx)
        self.recent_evictions = {
            k: v for k, v in self.recent_evictions.items() 
            if v["expires_at"] > frame_idx
        }
        evicted_masks = [v["mask"] for v in self.recent_evictions.values()]
        
        occupied_masks = established_masks + evicted_masks
        if not self.exploratory:
            occupied_masks = occupied_masks + transient_masks

        if occupied_masks:
            union = np.zeros((H, W), dtype=np.float32)
            for m in occupied_masks:
                union = np.maximum(union, np.squeeze(m).astype(np.float32))
            forward_mask = torch.as_tensor(union, device=device)  # [H, W]

        # Skip SAM1 if the frame is already almost fully covered.
        if forward_mask is not None and (forward_mask > 0).float().mean() > 0.99:
            return None

        # Sample grid prompts in uncovered regions.
        with torch.amp.autocast('cuda', enabled=False): 
            if forward_mask is not None:
                foreground_mask = (forward_mask > 0).float().unsqueeze(0).unsqueeze(0)  
                foreground_mask = F.interpolate(
                    foreground_mask,
                    scale_factor=1 / 16,
                    mode='bilinear',
                    antialias=True,
                )
                
                offset = 1 / (2 * n)
                points_one_side = torch.linspace(offset, 1 - offset, n, device=device)
                points_x = points_one_side.unsqueeze(0).repeat(n, 1)
                points_y = points_one_side.unsqueeze(1).repeat(1, n)
                points = torch.stack([points_x, points_y], dim=-1).unsqueeze(0)  
        
                points_label = F.grid_sample(
                    foreground_mask,
                    points * 2 - 1,
                    align_corners=False,
                ).view(-1)
        
                points = points.view(-1, 2)
                positive_points = points[points_label < 0.01].cpu().numpy()
        
                if len(positive_points) == 0:
                    return None
                out_default, out_s, out_m, out_l = self.segmenter.generate(
                    image_np, positive_points, None
                )
            else:
                offset = 1 / (2 * n)
                points_one_side = np.linspace(offset, 1 - offset, n)
                points_x = np.tile(points_one_side[np.newaxis, :], (n, 1))
                points_y = np.tile(points_one_side[:, np.newaxis], (1, n))
                grid_points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
                
                out_default, out_s, out_m, out_l = self.segmenter.generate(
                    image_np, grid_points, None
                )

        torch.cuda.empty_cache()
        
        def to_records(mdata):
            if mdata is None: return []
            mdata.to_numpy()
            items = list(mdata.items())
            if not items: return []
            num_masks = len(items[0][1])
            if num_masks == 0: return []
            return [{k: v[i] for k, v in items} for i in range(num_masks)]

        mask_default = to_records(out_default)
        mask_s = to_records(out_s)
        mask_m = to_records(out_m)
        mask_l = to_records(out_l)
        
        if self.adapt_num_points:
            # Use the default-mask count as a lightweight scene-complexity proxy.
            num_masks_generated = len(mask_default)
            self.grid_manager.update_and_get_grid(num_masks_generated)

        segmap = hierarchical_mask_merge((mask_s, mask_m, mask_l))

        if segmap is None or segmap.size == 0 or segmap.max() < 0:
            return None

        label_map_np = (segmap + 1).astype(np.int64)
        label_map_np = cv2.resize(
            label_map_np.astype(np.float32),
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

        return torch.as_tensor(label_map_np, dtype=torch.int64, device=device)
        
    @torch.inference_mode()
    def track(
        self,
        external_masks_per_frame=None,
        start_frame_idx=0,
        initial_next_obj_id=None,
        reset=True,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Main tracking entry point.

        Modes (in priority order):
          1. external_masks_per_frame provided  → offline mode
          2. self.segmenter is not None         → online mode (SAM1 grid prompts)
          3. neither                            → pure SAM2 propagation
        """
        if reset:
            self.reset_state()
        self._open_behavior_log()
        try:
            self._bootstrap(start_frame_idx, initial_next_obj_id, external_masks_per_frame)

            external_mask_keys = (
                list(external_masks_per_frame.keys())
                if external_masks_per_frame is not None
                else None
            )

            current_frame = start_frame_idx

            with tqdm(total=self._num_frames, desc="Savvy tracking") as pbar:
                pbar.update(start_frame_idx)

                while current_frame < self._num_frames:
                    new_obj_found = False

                    with _suppress_tqdm():
                        for frame_idx, obj_ids, video_res_masks in self.predictor.propagate_in_video(
                            self.inference_state,
                            start_frame_idx=current_frame,
                        ):
                            frame_started_at = time.perf_counter()
                            self._start_behavior_frame()
                            propagated_mask_count = len(obj_ids)
                            backbone_out = self._get_backbone_out(frame_idx)

                            self._store_results(
                                frame_idx, obj_ids, video_res_masks, backbone_out
                            )
                            pbar.update(1)

                            self._update_miss_streaks(frame_idx, backbone_out)
                            self._apply_suppression(frame_idx, backbone_out)

                            # The bootstrap frame is already registered.
                            if frame_idx == start_frame_idx and current_frame == start_frame_idx:
                                self._write_behavior_frame(
                                    frame_idx, frame_started_at, propagated_mask_count
                                )
                                continue

                            # Merge transient tracks that agree with established tracks.
                            if self._handle_merges(frame_idx, obj_ids):
                                self._write_behavior_frame(
                                    frame_idx, frame_started_at, propagated_mask_count
                                )
                                current_frame = frame_idx + 1
                                new_obj_found = True
                                break

                            # Promote stable transients or discard expired ones.
                            if self._handle_promotions(frame_idx):
                                self._write_behavior_frame(
                                    frame_idx, frame_started_at, propagated_mask_count
                                )
                                current_frame = frame_idx + 1
                                new_obj_found = True
                                break

                            # Keep SAM2 memory bounded on long videos.
                            self._prune_memory(frame_idx)
                            if self._enforce_memory_limit(frame_idx):
                                self._write_behavior_frame(
                                    frame_idx, frame_started_at, propagated_mask_count
                                )
                                current_frame = frame_idx + 1
                                new_obj_found = True
                                break

                            # Query for new objects at the segmenter stride.
                            if frame_idx % self.segmenter_stride == 0:
                                if self._handle_new_objects(
                                    frame_idx,
                                    video_res_masks,
                                    external_masks_per_frame,
                                    external_mask_keys,
                                ):
                                    self._write_behavior_frame(
                                        frame_idx, frame_started_at, propagated_mask_count
                                    )
                                    current_frame = frame_idx + 1
                                    new_obj_found = True
                                    break

                            self._write_behavior_frame(
                                frame_idx, frame_started_at, propagated_mask_count
                            )

                    if not new_obj_found:
                        break
        finally:
            self._close_behavior_log()

        self.all_results = backfill_transient_gaps(
            self.all_results, self.transient_mask_buffer
        )
        return self.all_results

    def _bootstrap(self, 
                   start_frame_idx, 
                   initial_next_obj_id, 
                   external_masks_per_frame,
                   num_points_per_side=None):
        
        if initial_next_obj_id is None:
            existing_ids = self.inference_state["obj_ids"]
            self.next_obj_id = (max(existing_ids) + 1) if existing_ids else 0
        else:
            self.next_obj_id = initial_next_obj_id
    
        if external_masks_per_frame is not None:
            external_mask_keys = list(external_masks_per_frame.keys())
            initial_external = external_masks_per_frame.get(
                external_mask_keys[start_frame_idx], None
            )
            if initial_external is not None:
                self.next_obj_id = _bootstrap_initial_frame(
                    self.predictor,
                    self.inference_state,
                    initial_external,
                    start_frame_idx,
                    self.next_obj_id,
                )
    
        elif self.segmenter is not None:
            # Online mode: query SAM1 on the first frame.
            initial_external = self._query_segmenter(start_frame_idx, num_points_per_side)
            if initial_external is not None:
                self.next_obj_id = _bootstrap_initial_frame(
                    self.predictor,
                    self.inference_state,
                    initial_external,
                    start_frame_idx,
                    self.next_obj_id,
                )
    
        if self.predictor._get_obj_num(self.inference_state) == 0:
            raise RuntimeError(
                f"No objects at start_frame_idx={start_frame_idx}. "
                f"Provide external_masks, a segmenter, or pre-register objects."
            )

    def _get_backbone_out(self, frame_idx):
        _, backbone_out = self.inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        return backbone_out

    def _store_results(self, frame_idx, obj_ids, video_res_masks, backbone_out):
        if frame_idx not in self.all_results:
            self.all_results[frame_idx] = {}

        for i, out_obj_id in enumerate(obj_ids):
            mask = (video_res_masks[i] > 0.0).cpu().numpy().astype(np.bool_)

            current_area = int(np.sum(mask))
            if current_area > self.max_historical_area.get(out_obj_id, 0):
                self.max_historical_area[out_obj_id] = current_area
                
            if self.buffer.is_transient(out_obj_id):
                self.transient_mask_buffer.setdefault(out_obj_id, {})[frame_idx] = mask
                # Warm up EMA during transient period
                track = self.buffer.transients.get(out_obj_id)
                if track is not None and np.squeeze(mask).any():
                    track.visible_frames += 1
                    area = int(np.squeeze(mask).sum())
                    track.max_area = max(track.max_area, area)

                if backbone_out is not None and np.squeeze(mask).any():
                    self.obj_location_feature.update(
                        obj_id=out_obj_id,
                        mask=mask,
                        backbone_out=backbone_out,
                        image_hw=self._image_hw,
                    )
            else:
                self.all_results[frame_idx][out_obj_id] = mask

    def _update_miss_streaks(self, frame_idx, backbone_out):
        for oid in list(self.established_miss_streak.keys()):
            mask = self.all_results.get(frame_idx, {}).get(oid)
            is_present = mask is not None and np.squeeze(mask).any()

            if is_present:
                if (
                    self.established_miss_streak[oid] >= self.buffer_size
                    and backbone_out is not None
                ):
                    squeezed = np.squeeze(mask)
                    ys, xs = np.where(squeezed)
                    mask_area = int(squeezed.sum())
                    current_descriptor = self.obj_location_feature._extract_descriptor(
                        backbone_out=backbone_out,
                        x1=int(xs.min()),
                        y1=int(ys.min()),
                        x2=int(xs.max()),
                        y2=int(ys.max()),
                        image_hw=self._image_hw,
                        mask_area=mask_area,
                    )
                    similarity = self.obj_location_feature.max_similarity(
                        oid, current_descriptor
                    )
                    if similarity is not None and similarity < self.memory_strength:
                        self.all_results[frame_idx].pop(oid, None)
                        self.established_miss_streak[oid] += 1
                        self.continuous_presence_streak[oid] = 0
                        continue

                # The appearance gate passed, or no appearance bank exists yet.
                self.established_miss_streak[oid] = 0
                self.continuous_presence_streak[oid] = (
                    self.continuous_presence_streak.get(oid, 0) + 1
                )
                self.obj_location_feature.update(
                    obj_id=oid,
                    mask=mask,
                    backbone_out=backbone_out,
                    image_hw=self._image_hw,
                )

            else:
                prev_streak = self.established_miss_streak.get(oid, 0)
                self.established_miss_streak[oid] = prev_streak + 1
                self.continuous_presence_streak[oid] = 0

                # Commit the active appearance EMA once the object is absent long enough.
                if prev_streak + 1 == self.buffer_size:
                    self.obj_location_feature.commit(oid)

    def _apply_suppression(self, frame_idx, backbone_out):
        established_ids_set = set(
            oid for oid in self.all_results.get(frame_idx, {})
            if not self.buffer.is_transient(oid)
        )

        if len(established_ids_set) <= 1:
            return

        consistency_scores: Dict[int, float] = {}
        if backbone_out is not None:
            for oid in established_ids_set:
                mask = self.all_results[frame_idx].get(oid)
                if mask is None:
                    continue
                squeezed = np.squeeze(mask)
                if not squeezed.any():
                    continue
                ys, xs = np.where(squeezed)
                mask_area = int(squeezed.sum())
                current_descriptor = self.obj_location_feature._extract_descriptor(
                    backbone_out=backbone_out,
                    x1=int(xs.min()),
                    y1=int(ys.min()),
                    x2=int(xs.max()),
                    y2=int(ys.max()),
                    image_hw=self._image_hw,
                    mask_area=mask_area,
                )
                score = self.obj_location_feature.max_similarity(oid, current_descriptor)
                if score is not None:
                    consistency_scores[oid] = score

        suppressed_ids = apply_seniority_suppression_to_frame(
            self.all_results[frame_idx],
            established_ids=established_ids_set,
            consistency_scores=consistency_scores,
            continuous_presence_streaks=self.continuous_presence_streak,
            iou_threshold=0.5,
            margin=self.margin,
        )

        for oid in suppressed_ids:
            self.all_results[frame_idx].pop(oid, None)
            self.established_miss_streak[oid] = (
                self.established_miss_streak.get(oid, 0) + 1
            )
            self.continuous_presence_streak[oid] = 0

    def _handle_merges(self, frame_idx, obj_ids) -> bool:
        established_ids = [
            oid for oid in obj_ids
            if not self.buffer.is_transient(oid)
            and self.all_results.get(frame_idx, {}).get(oid) is not None
            and np.squeeze(self.all_results[frame_idx][oid]).any()
        ]
        transient_ids = [
            oid for oid in obj_ids
            if self.buffer.is_transient(oid)
            and self.transient_mask_buffer.get(oid, {}).get(frame_idx) is not None
            and self.transient_mask_buffer[oid][frame_idx].any()
        ]

        if not (transient_ids and established_ids):
            return False

        _, all_pred_masks, _ = _collect_obj_ptrs_and_masks(
            self.inference_state, obj_ids, frame_idx
        )

        merges = self.buffer.check_agreements(
            frame_idx=frame_idx,
            established_obj_ids=established_ids,
            transient_obj_ids=transient_ids,
            all_pred_masks=all_pred_masks,
        )

        if not merges:
            return False

        for transient_id, established_id in merges:
            _merge_transient_into_established(
                self.predictor, self.inference_state,
                transient_obj_id=transient_id,
                established_obj_id=established_id,
            )
            self.buffer.remove_transient(transient_id)

            for f, frame_result in self.all_results.items():
                if transient_id in frame_result:
                    if established_id not in frame_result:
                        frame_result[established_id] = frame_result.pop(transient_id)
                    else:
                        frame_result.pop(transient_id)

            for f, mask in self.transient_mask_buffer.pop(transient_id, {}).items():
                if f not in self.all_results:
                    continue
                if not mask.any():
                    continue
                existing = self.all_results[f].get(established_id)
                if existing is None or not existing.any():
                    self.all_results[f][established_id] = mask
                else:
                    self.all_results[f][established_id] = existing | mask

        return True

    def _handle_promotions(self, frame_idx) -> bool:
        # Apply at most one queued part-whole absorption per restart.
        if self.pending_absorptions:
            whole_id, part_id = self.pending_absorptions.pop(0)

            if (
                whole_id in self.inference_state["obj_id_to_idx"]
                and part_id in self.inference_state["obj_id_to_idx"]
            ):
                for f, frame_result in self.all_results.items():
                    if part_id not in frame_result:
                        continue
                    src = frame_result.pop(part_id)
                    if whole_id not in frame_result:
                        frame_result[whole_id] = src
                    else:
                        frame_result[whole_id] = frame_result[whole_id] | src

                self._remove_object(part_id)

            return True

        expired = self.buffer.get_expired_transients(frame_idx)

        for obj_id in expired["promote"]:
            buffer_obj = self.buffer
            buffer_obj.promote_to_established(obj_id)

            # Backfill frames where the transient was temporarily suppressed.
            for f, mask in self.transient_mask_buffer.pop(obj_id, {}).items():
                if f not in self.all_results:
                    continue
                if obj_id in self.all_results[f]:
                    continue
                if mask.any():
                    self.all_results[f][obj_id] = mask

            # Register the promoted object as established.
            self.established_miss_streak[obj_id] = 0
            self.continuous_presence_streak[obj_id] = self.buffer_size

            # Absorb established part tracks that are contained in the new whole track.
            established_ids = [
                oid for oid in self.established_miss_streak
                if oid != obj_id
                and not self.buffer.is_transient(oid)
            ]
            absorb_targets = _find_contained_parts(
                whole_id=obj_id,
                frame_idx=frame_idx,
                established_ids=established_ids,
                all_results=self.all_results,
                containment_threshold=0.7,
                min_area_ratio=1.5,
            )

            if absorb_targets:
                self.pending_absorptions.extend(
                    (obj_id, p) for p in absorb_targets[1:]
                )
                part_id = absorb_targets[0]

                for f, frame_result in self.all_results.items():
                    if part_id not in frame_result:
                        continue
                    src = frame_result.pop(part_id)
                    if obj_id not in frame_result:
                        frame_result[obj_id] = src
                    else:
                        frame_result[obj_id] = frame_result[obj_id] | src

                self._remove_object(part_id)
                return True

        if expired["discard"]:
            for obj_id in expired["discard"]:
                self._remove_object(obj_id)
                for frame_result in self.all_results.values():
                    frame_result.pop(obj_id, None)
            return True

        return False

    def _prune_memory(self, frame_idx):
        if (
            self.prune_memory_every_n_frames > 0
            and frame_idx % self.prune_memory_every_n_frames == 0
        ):
            _prune_old_non_cond_outputs(
                self.inference_state,
                current_frame_idx=frame_idx,
                num_maskmem=self.predictor.num_maskmem,
                stride=self.predictor.memory_temporal_stride_for_eval,
            )
            _strip_pred_masks_from_old_frames(
                self.inference_state,
                current_frame_idx=frame_idx,
            )

    def _handle_new_objects(
        self,
        frame_idx,
        video_res_masks,
        external_masks_per_frame,
        external_mask_keys,
        num_points_per_side=None
    ) -> bool:
        external_masks = None
    
        if external_masks_per_frame is not None:
            external_masks = external_masks_per_frame.get(
                external_mask_keys[frame_idx], None
            )
        elif self.segmenter is not None:
            external_masks = self._query_segmenter(frame_idx, num_points_per_side=None)
    
        if external_masks is None:
            return False

        self._behavior_frame_stats["segmenter_invoked"] = 1
        self._behavior_frame_stats["segmenter_candidate_mask_count"] = (
            self._count_segmenter_candidates(external_masks)
        )
    
        extra_suppress = [
            m for t_id, t_frame_masks in self.transient_mask_buffer.items()
            for f, m in t_frame_masks.items() if f == frame_idx
        ]
    
        new_objects, self.next_obj_id = new_mask_fn(
            frame_idx=frame_idx,
            video_res_masks=video_res_masks,
            external_masks=external_masks,
            inference_state=self.inference_state,
            next_obj_id=self.next_obj_id,
            survival_threshold=self.survival_threshold,
            extra_suppress_masks=extra_suppress,
        )
    
        if not new_objects:
            return False

        self._behavior_frame_stats["newly_discovered_mask_count"] = len(new_objects)
    
        for new_obj_id, surviving_mask in new_objects:
            self.predictor.add_new_mask(
                self.inference_state,
                frame_idx=frame_idx,
                obj_id=new_obj_id,
                mask=surviving_mask,
            )
            self.buffer.add_transient(
                obj_id=new_obj_id,
                entry_frame=frame_idx,
            )
    
        _flush_new_obj_temp_outputs(self.predictor, self.inference_state)
        return True
    
    def _enforce_memory_limit(self, frame_idx):
        """
        Circuit breaker for SAM2 VRAM. Prunes based on the MAXIMUM area an 
        object has ever achieved to protect objects far away in perspective.
        """
        active_ids = list(self.inference_state.get("obj_id_to_idx", {}).keys())
        
        if len(active_ids) <= self.max_active_objects:
            return False

        prunable_ids = [oid for oid in active_ids if not self.buffer.is_transient(oid)]

        if len(prunable_ids) == 0:
            prunable_ids = active_ids

        print(f"  [!] VRAM Emergency: {len(active_ids)} active tracks at frame {frame_idx}. Pruning noise...")

        # Prefer pruning tracks that have only ever been small.
        sorted_by_historical_size = sorted(
            prunable_ids, 
            key=lambda oid: self.max_historical_area.get(oid, 0)
        )

        dropped_count = 0
        target_drop_count = len(active_ids) - self.max_active_objects

        for oid in sorted_by_historical_size:
            max_area = self.max_historical_area.get(oid, 0)
            
            if dropped_count >= target_drop_count:
                break

            # Temporarily block the removed region to avoid immediate rediscovery.
            last_mask = self.all_results.get(frame_idx, {}).get(oid)
            if last_mask is not None:
                # Keep the space blocked for three segmenter pulses.
                self.recent_evictions[oid] = {
                    "mask": last_mask, 
                    "expires_at": frame_idx + (self.segmenter_stride * 3)
                }
                
            # Drop the selected track from SAM2 memory and bookkeeping.
            if max_area < self.min_area_to_keep:
                self._remove_object(oid)
                dropped_count += 1
            else:
                self._remove_object(oid)
                dropped_count += 1

        if dropped_count > 0:
            print(f"  [~] Dropped {dropped_count} established objects. Active SAM2 tracks: {len(self.inference_state.get('obj_id_to_idx', {}))}.")
        return dropped_count > 0
    
    def _remove_object(self, obj_id):
        """Centralised cleanup for any removed object."""
        self.predictor.remove_object(
            self.inference_state,
            obj_id=obj_id,
            strict=False,
            need_output=False,
        )
        self.buffer.remove_transient(obj_id)
        self.transient_mask_buffer.pop(obj_id, None)
        self.established_miss_streak.pop(obj_id, None)
        self.continuous_presence_streak.pop(obj_id, None)
        self.obj_location_feature.remove(obj_id)
        self.max_historical_area.pop(obj_id, None)
