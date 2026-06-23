from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import box_cxcywh_to_xyxy, generalized_box_iou


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / max(num_masks, 1.0)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / max(num_masks, 1.0)


class SetCriterion(nn.Module):
    """Loss for depth-only VQ detector.

    GT masks are used only here and in Hungarian matching. They are never model input.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: dict[str, float],
        eos_coef: float = 0.1,
        cad_codebook: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.matcher = matcher
        self.weight_dict = weight_dict
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)
        if cad_codebook is not None:
            self.register_buffer("cad_codebook", F.normalize(cad_codebook.float(), dim=-1))
        else:
            self.register_buffer("cad_codebook", torch.empty(0, 1))

    def _get_src_permutation_idx(self, indices: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]) if indices else torch.empty(0, dtype=torch.long)
        src_idx = torch.cat([src for (src, _) in indices]) if indices else torch.empty(0, dtype=torch.long)
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]) if indices else torch.empty(0, dtype=torch.long)
        tgt_idx = torch.cat([tgt for (_, tgt) in indices]) if indices else torch.empty(0, dtype=torch.long)
        return batch_idx, tgt_idx

    def loss_labels(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, Any]], indices: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
        src_logits = outputs["pred_logits"]
        device = src_logits.device
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)]).to(device) if idx[0].numel() else torch.empty(0, dtype=torch.long, device=device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.long, device=device)
        if idx[0].numel():
            target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, Any]], indices: list[tuple[torch.Tensor, torch.Tensor]], num_boxes: float) -> dict[str, torch.Tensor]:
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_boxes.device)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / max(num_boxes, 1.0)
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou.sum() / max(num_boxes, 1.0)}

    def loss_masks(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, Any]], indices: list[tuple[torch.Tensor, torch.Tensor]], num_masks: float) -> dict[str, torch.Tensor]:
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = outputs["pred_masks"].sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}
        src_masks = outputs["pred_masks"][idx]
        target_masks = torch.cat([t["masks"][i] for t, (_, i) in zip(targets, indices)], dim=0).to(src_masks.device)
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        loss_mask = sigmoid_ce_loss(src_masks.flatten(1), target_masks.flatten(1), num_masks)
        loss_dice = dice_loss(src_masks, target_masks, num_masks)
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def loss_cad(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, Any]], indices: list[tuple[torch.Tensor, torch.Tensor]], num_boxes: float) -> dict[str, torch.Tensor]:
        if "pred_cad_logits" not in outputs:
            zero = outputs["pred_obj_emb"].sum() * 0.0
            return {"loss_cad_ce": zero, "loss_cad_align": zero}
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            zero = outputs["pred_cad_logits"].sum() * 0.0
            return {"loss_cad_ce": zero, "loss_cad_align": zero}
        cad_ids = torch.cat([t["cad_ids"][i] for t, (_, i) in zip(targets, indices)], dim=0).to(outputs["pred_cad_logits"].device)
        valid = cad_ids >= 0
        if not valid.any():
            zero = outputs["pred_cad_logits"].sum() * 0.0
            return {"loss_cad_ce": zero, "loss_cad_align": zero}
        b_idx, q_idx = idx
        b_idx = b_idx.to(outputs["pred_cad_logits"].device)[valid]
        q_idx = q_idx.to(outputs["pred_cad_logits"].device)[valid]
        cad_ids_valid = cad_ids[valid]
        logits = outputs["pred_cad_logits"][b_idx, q_idx]
        loss_cad_ce = F.cross_entropy(logits, cad_ids_valid, reduction="sum") / max(float(valid.sum().item()), 1.0)

        emb = outputs["pred_obj_emb"][b_idx, q_idx]
        if self.cad_codebook.numel() > 0:
            gt_z = self.cad_codebook.to(emb.device)[cad_ids_valid]
            loss_align = (1 - F.cosine_similarity(emb, gt_z, dim=-1)).sum() / max(float(valid.sum().item()), 1.0)
        else:
            loss_align = emb.sum() * 0.0
        return {"loss_cad_ce": loss_cad_ce, "loss_cad_align": loss_align}

    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes_t = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs["pred_logits"].device)
        # Distributed training hook can be added here if needed.
        num_boxes_float = max(float(num_boxes_t.item()), 1.0)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes_float))
        losses.update(self.loss_masks(outputs, targets, indices, num_boxes_float))
        losses.update(self.loss_cad(outputs, targets, indices, num_boxes_float))

        weighted = {}
        total = 0.0
        for k, v in losses.items():
            w = self.weight_dict.get(k, 0.0)
            weighted[k] = v
            if w != 0:
                total = total + w * v
        weighted["loss_total"] = total
        return weighted
