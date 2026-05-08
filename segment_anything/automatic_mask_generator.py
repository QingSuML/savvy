# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)

class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        All processing, including area and quality score calculation, is done on GPU.
        Returns a 4-tuple of MaskData: (default, small, medium, large).
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

    @torch.no_grad()
    def generate(
        self,
        image: np.ndarray,
        positive_points: Optional[np.ndarray] = None,
        negative_points: Optional[np.ndarray] = None,
    ) -> Tuple[MaskData, MaskData, MaskData, MaskData]:
        """
        Generates masks for the image. 
        Returns (all_masks, small_masks, medium_masks, large_masks).
        """
        return self._generate_masks(image, positive_points, negative_points)

    def _generate_masks(
        self,
        image: np.ndarray,
        positive_points: Optional[np.ndarray],
        negative_points: Optional[np.ndarray]
    ) -> Tuple[MaskData, MaskData, MaskData, MaskData]:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(orig_size, self.crop_n_layers, self.crop_overlap_ratio)

        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size, positive_points, negative_points)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1 and ("boxes" in data._stats):
            scores = 1.0 / box_area(data["crop_boxes"])
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores.to(data["boxes"].device),
                torch.zeros_like(data["boxes"][:, 0]),
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        # Split by scale_id (0: small, 1: medium, 2: large)
        scale_id = data._stats.get("scale_id", torch.tensor([], device=self.predictor.device))

        data_s = copy.deepcopy(data)
        data_m = copy.deepcopy(data)
        data_l = copy.deepcopy(data)

        if scale_id.numel() > 0:
            data_s.filter(scale_id == 0)
            data_m.filter(scale_id == 1)
            data_l.filter(scale_id == 2)
        else:
            data_s, data_m, data_l = MaskData(), MaskData(), MaskData()

        return data, data_s, data_m, data_l

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
        positive_points: Optional[np.ndarray],
        negative_points: Optional[np.ndarray],
    ) -> MaskData:
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        points_scale = np.array(cropped_im_size)[None, ::-1]

        if positive_points is not None:
            points_for_image = positive_points * points_scale
        else:
            points_for_image = self.point_grids[crop_layer_idx] * points_scale

        scaled_neg_points = negative_points * points_scale if negative_points is not None else None

        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, scaled_neg_points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data

        self.predictor.reset_image()

        if "boxes" not in data._stats or data["boxes"].numel() == 0:
            return data

        # Remove duplicates within this crop
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Map back to original image coordinates
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["boxes"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        negative_points: Optional[np.ndarray],
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        if negative_points is not None:
            neg_repeated = np.repeat(negative_points[None, :, :], points.shape[0], axis=0)
            points_combined = np.concatenate([points[:, None, :], neg_repeated], axis=1)
            transformed_points = self.predictor.transform.apply_coords(points_combined, im_size)
            in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
            in_labels = torch.zeros((in_points.shape[0], in_points.shape[1]), dtype=torch.int, device=in_points.device)
            in_labels[:, 0] = 1 # Grid/Positive point is 1, rest are 0
        else:
            transformed_points = self.predictor.transform.apply_coords(points, im_size)
            in_points = torch.as_tensor(transformed_points, device=self.predictor.device)[:, None, :]
            in_labels = torch.ones((in_points.shape[0], 1), dtype=torch.int, device=in_points.device)

        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points, in_labels, multimask_output=True, return_logits=True
        )
        B, K, H, W = masks.shape

        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points, device=self.predictor.device).repeat_interleave(K, dim=0),
            scale_id=torch.arange(K, device=self.predictor.device).repeat(B)
        )
        del masks

        if self.pred_iou_thresh > 0.0:
            data.filter(data["iou_preds"] > self.pred_iou_thresh)

        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            data.filter(data["stability_score"] >= self.stability_score_thresh)

        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # GPU-based area and quality score
        data["area"] = data["masks"].sum(dim=(1, 2))
        data["quality_score"] = (1 + torch.log1p(data["area"].float())) * data["iou_preds"]

        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        data.filter(keep_mask)

        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)

        return data