from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TransientTrack:
    obj_id: int
    entry_frame: int

    best_match_id: Optional[int] = None
    agreement_hits: int = 0
    best_iou: float = 0.0

    visible_frames: int = 0
    max_area: int = 0

class HandshakeBuffer:
    def __init__(
        self,
        buffer_size: int = 30,
        iou_threshold: float = 0.5,
        min_agreement_hits: int = 2,
        min_visible_frames_for_promotion: int = 3,
        min_max_area_for_promotion: int = -1,
        min_max_area_ratio_for_promotion: float = -1.0,
        image_hw: Optional[Tuple[int, int]] = None,
    ):
        self.buffer_size = buffer_size
        self.iou_threshold = iou_threshold
        self.min_agreement_hits = min_agreement_hits
        
        self.min_visible_frames_for_promotion = min_visible_frames_for_promotion
        self.min_max_area_for_promotion = min_max_area_for_promotion
        self.min_max_area_ratio_for_promotion = min_max_area_ratio_for_promotion
        self.image_hw = image_hw
        self.transients: Dict[int, TransientTrack] = {}

    def add_transient(self, obj_id: int, entry_frame: int):
        self.transients[obj_id] = TransientTrack(obj_id=obj_id, entry_frame=entry_frame)

    def is_transient(self, obj_id: int) -> bool:
        return obj_id in self.transients
    
    def _passes_short_track_promotion(self, track: TransientTrack) -> bool:
        """
        Decide whether a transient has enough visible evidence to be promoted
        at buffer expiry, even if it was short-lived.

        Rules:
        - must have at least min_visible_frames_for_promotion visible frames
        - if min_max_area_for_promotion >= 0, require max_area >= threshold
        - if min_max_area_ratio_for_promotion >= 0, require max_area/image_area >= ratio
        - if both area thresholds are disabled (<0), only visible_frames is enforced
        """
        if track.visible_frames < self.min_visible_frames_for_promotion:
            return False

        if self.min_max_area_for_promotion >= 0:
            if track.max_area < self.min_max_area_for_promotion:
                return False

        if self.min_max_area_ratio_for_promotion >= 0:
            if self.image_hw is None:
                return False
            H, W = self.image_hw
            image_area = max(H * W, 1)
            if (track.max_area / image_area) < self.min_max_area_ratio_for_promotion:
                return False

        return True

    def get_expired_transients(self, current_frame: int) -> Dict[str, List[int]]:
        """
        Resolve expired transient tracks at the END of the handshake window.

        Decision order:
        1. If duplicate evidence is strong enough -> discard
        2. Else if visible-period evidence is strong enough -> promote
        3. Else -> discard
        """
        promote = []
        discard = []

        for obj_id, track in self.transients.items():
            if current_frame - track.entry_frame < self.buffer_size:
                continue

            # Strong repeated duplicate evidence
            if (
                track.best_match_id is not None
                and track.agreement_hits >= self.min_agreement_hits
            ):
                discard.append(obj_id)
                continue

            # Distinct enough and real enough to keep
            if self._passes_short_track_promotion(track):
                promote.append(obj_id)
            else:
                discard.append(obj_id)

        return {"promote": promote, "discard": discard}

    def promote_to_established(self, obj_id: int):
        self.transients.pop(obj_id, None)

    def remove_transient(self, obj_id: int):
        self.transients.pop(obj_id, None)

    def check_agreements(
        self,
        frame_idx,
        established_obj_ids,
        transient_obj_ids,
        all_pred_masks,
    ):
        merges = []

        for t_id in transient_obj_ids:
            track = self.transients.get(t_id)
            if track is None:
                continue

            t_mask = all_pred_masks.get(t_id)
            if t_mask is None:
                continue

            t_bin = (t_mask > 0).float()
            t_area = t_bin.sum().item()
            if t_area == 0:
                continue

            best_iou = 0.0
            best_e_id = None

            for e_id in established_obj_ids:
                e_mask = all_pred_masks.get(e_id)
                if e_mask is None:
                    continue

                e_bin = (e_mask > 0).float()
                e_area = e_bin.sum().item()
                if e_area == 0:
                    continue

                intersection = (t_bin * e_bin).sum().item()
                if intersection == 0:
                    continue

                union = ((t_bin + e_bin) > 0).float().sum().item()
                iou = intersection / union if union > 0 else 0.0

                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_e_id = e_id

            if best_e_id is None:
                continue

            if track.best_match_id is None:
                track.best_match_id = best_e_id
                track.agreement_hits = 1
                track.best_iou = best_iou
            elif track.best_match_id == best_e_id:
                track.agreement_hits += 1
                track.best_iou = max(track.best_iou, best_iou)
            else:
                # Competing matches: switch only if clearly stronger
                if best_iou > track.best_iou * 1.25:
                    track.best_match_id = best_e_id
                    track.agreement_hits = 1
                    track.best_iou = best_iou

            if (
                track.best_match_id is not None
                and track.agreement_hits >= self.min_agreement_hits
            ):
                merges.append((t_id, track.best_match_id))

        return merges
