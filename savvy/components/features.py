import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class ObjectLocationFeature:
    """
    Maintains a per-object appearance memory bank of Hiera FPN pooled features.
    Each entry in the bank corresponds to one contiguous appearance period.

    Within each appearance period, a local EMA is maintained. When a period
    ends (miss streak crosses buffer_size), the local EMA is frozen and
    committed to the bank as a new entry.

    Feature extraction:
      - For each of the 4 FPN levels, the tight mask bbox is expanded by
        box_scale (e.g. 1.5x), scaled to that level's resolution, clamped
        to bounds, and global-average-pooled.
      - The 4 pooled vectors are concatenated → one descriptor per frame.
      - For large masks (area > 15% of image), tight bbox is used instead
        of dilated to avoid the descriptor becoming a global scene descriptor.

    Comparison:
      - max cosine similarity across all bank entries is used for gating
        and seniority suppression.

    Args:
        box_scale:      bbox dilation factor for small/medium masks
        ema_alpha:      EMA decay for within-period updates
        max_bank_size:  max number of appearance periods to retain per object
                        (None = unbounded, FIFO eviction when exceeded)
    """

    def __init__(
        self,
        box_scale: float = 1.5,
        ema_alpha: float = 0.9,
        max_bank_size: Optional[int] = None,
    ):
        self.box_scale = box_scale
        self.ema_alpha = ema_alpha
        self.max_bank_size = max_bank_size

        # Committed appearance vectors per object, stored on CPU.
        self._bank: Dict[int, List[torch.Tensor]] = {}

        # In-progress EMA for each active appearance period.
        self._active_ema: Dict[int, torch.Tensor] = {}

        # Active appearance-period flags.
        self._is_active: Dict[int, bool] = {}

    def update(
        self,
        obj_id: int,
        mask: np.ndarray,
        backbone_out: dict,
        image_hw: Tuple[int, int],
    ) -> None:
        """
        Called on every confirmed present frame (gate passed or normal present).
        Updates the active EMA for the current appearance period.

        Args:
            obj_id:       established object id
            mask:         bool numpy array, shape (H, W) or variations
            backbone_out: inference_state["cached_features"][frame_idx][1]
            image_hw:     (H, W) of the original image
        """
        mask_2d = np.squeeze(mask)
        ys, xs = np.where(mask_2d)
        if len(ys) == 0:
            return

        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        mask_area = int(mask_2d.sum())

        descriptor = self._extract_descriptor(
            backbone_out=backbone_out,
            x1=x1, y1=y1, x2=x2, y2=y2,
            image_hw=image_hw,
            mask_area=mask_area,
        )

        if obj_id not in self._active_ema:
            # First frame of a new appearance period.
            self._active_ema[obj_id] = descriptor
        else:
            self._active_ema[obj_id] = (
                self.ema_alpha * self._active_ema[obj_id]
                + (1.0 - self.ema_alpha) * descriptor
            )

        self._is_active[obj_id] = True

    def commit(self, obj_id: int) -> None:
        """
        Called when an object's appearance period ends — i.e. when its miss
        streak crosses buffer_size. Freezes the active EMA and commits it
        to the bank. Clears the active EMA so the next appearance period
        starts fresh.
        """
        if obj_id not in self._active_ema:
            return
        if not self._is_active.get(obj_id, False):
            return

        vec = self._active_ema.pop(obj_id)
        self._is_active[obj_id] = False

        if obj_id not in self._bank:
            self._bank[obj_id] = []

        self._bank[obj_id].append(vec)

        # Evict the oldest appearance period if the bank exceeds max size.
        if self.max_bank_size is not None:
            while len(self._bank[obj_id]) > self.max_bank_size:
                self._bank[obj_id].pop(0)

    def max_similarity(
        self,
        obj_id: int,
        descriptor: torch.Tensor,
    ) -> Optional[float]:
        """
        Max cosine similarity between descriptor and all entries in obj_id's bank.
        Also considers the active EMA if currently in an appearance period.
        Returns None if no entries exist yet.
        """
        candidates = list(self._bank.get(obj_id, []))
        if self._is_active.get(obj_id, False) and obj_id in self._active_ema:
            candidates.append(self._active_ema[obj_id])

        if not candidates:
            return None

        stacked = torch.stack(candidates, dim=0)          # (N, C)
        query   = descriptor.unsqueeze(0).expand_as(stacked)  # (N, C)
        sims    = torch.nn.functional.cosine_similarity(stacked, query, dim=1)  # (N,)
        return sims.max().item()

    def get_active_ema(self, obj_id: int) -> Optional[torch.Tensor]:
        """Return current active EMA for obj_id, or None if not in active period."""
        if self._is_active.get(obj_id, False):
            return self._active_ema.get(obj_id, None)
        return None

    def get_bank(self, obj_id: int) -> List[torch.Tensor]:
        """Return all committed appearance vectors for obj_id."""
        return self._bank.get(obj_id, [])

    def remove(self, obj_id: int) -> None:
        """Remove all state for obj_id (on removal or absorption)."""
        self._bank.pop(obj_id, None)
        self._active_ema.pop(obj_id, None)
        self._is_active.pop(obj_id, None)

    def _extract_descriptor(
        self,
        backbone_out: dict,
        x1: int, y1: int, x2: int, y2: int,
        image_hw: Tuple[int, int],
        mask_area: Optional[int] = None,
    ) -> torch.Tensor:

        H_img, W_img = image_hw
        image_area = H_img * W_img
        fpn = backbone_out["backbone_fpn"]

        if mask_area is not None and mask_area / image_area > 0.15:
            x1e, y1e, x2e, y2e = x1, y1, x2, y2
        else:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = (x2 - x1) * self.box_scale
            h  = (y2 - y1) * self.box_scale
            x1e = cx - w / 2.0
            x2e = cx + w / 2.0
            y1e = cy - h / 2.0
            y2e = cy + h / 2.0

        pooled = []
        for feat in fpn:
            _, C_l, H_l, W_l = feat.shape

            scale_x = W_l / W_img
            scale_y = H_l / H_img

            fx1 = max(0,   int(math.floor(x1e * scale_x)))
            fx2 = min(W_l, int(math.ceil (x2e * scale_x)))
            fy1 = max(0,   int(math.floor(y1e * scale_y)))
            fy2 = min(H_l, int(math.ceil (y2e * scale_y)))

            fx2 = max(fx2, fx1 + 1)
            fy2 = max(fy2, fy1 + 1)

            crop = feat[0, :, fy1:fy2, fx1:fx2]
            pooled_vec = crop.mean(dim=(-2, -1))
            pooled.append(pooled_vec.cpu().float())

        return torch.cat(pooled, dim=0)
