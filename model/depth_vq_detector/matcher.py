from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import box_cxcywh_to_xyxy, generalized_box_iou

try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except Exception:  # pragma: no cover
    _linear_sum_assignment = None


def _greedy_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback if scipy is unavailable. Not exact Hungarian, but useful for smoke tests."""
    cost = cost.clone()
    q, t = cost.shape
    rows = []
    cols = []
    used_r = set()
    used_c = set()
    for _ in range(min(q, t)):
        flat_idx = torch.argmin(cost).item()
        r = flat_idx // t
        c = flat_idx % t
        if r in used_r or c in used_c:
            cost[r, c] = float("inf")
            continue
        rows.append(r)
        cols.append(c)
        used_r.add(r)
        used_c.add(c)
        cost[r, :] = float("inf")
        cost[:, c] = float("inf")
    return torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)


def batch_dice_cost(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pairwise Dice cost. inputs [Q,HW] are probabilities, targets [T,HW] are 0/1."""
    numerator = 2 * torch.einsum("qc,tc->qt", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1.0) / (denominator + 1.0)


def batch_sigmoid_ce_cost(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pairwise BCE cost. logits [Q,HW], targets [T,HW]."""
    q, hw = logits.shape
    t = targets.shape[0]
    if t == 0:
        return logits.new_zeros((q, 0))
    logits_exp = logits[:, None, :].expand(q, t, hw)
    targets_exp = targets[None, :, :].expand(q, t, hw)
    return F.binary_cross_entropy_with_logits(logits_exp, targets_exp, reduction="none").mean(-1)


class HungarianMatcher(nn.Module):
    """Assign predicted object queries to GT instances."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cost_mask: float = 2.0,
        cost_dice: float = 2.0,
        cost_cad: float = 0.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_cad = cost_cad

    @torch.no_grad()
    def forward(self, outputs: dict[str, torch.Tensor], targets: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]
        out_masks = outputs.get("pred_masks")
        out_cad = outputs.get("pred_cad_logits")

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"].to(out_prob.device)
            tgt_bbox = targets[b]["boxes"].to(out_bbox.device)
            num_tgt = tgt_ids.shape[0]
            if num_tgt == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox))

            total = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

            if out_masks is not None and (self.cost_mask > 0 or self.cost_dice > 0):
                pred_m = out_masks[b]
                tgt_m = targets[b]["masks"].to(pred_m.device).float()
                tgt_m = F.interpolate(tgt_m[:, None], size=pred_m.shape[-2:], mode="nearest")[:, 0]
                pred_flat = pred_m.flatten(1)
                tgt_flat = tgt_m.flatten(1)
                if self.cost_mask > 0:
                    total = total + self.cost_mask * batch_sigmoid_ce_cost(pred_flat, tgt_flat)
                if self.cost_dice > 0:
                    total = total + self.cost_dice * batch_dice_cost(pred_flat.sigmoid(), tgt_flat)

            if out_cad is not None and self.cost_cad > 0:
                cad_ids = targets[b].get("cad_ids")
                if cad_ids is not None:
                    cad_ids = cad_ids.to(out_cad.device)
                    valid = cad_ids >= 0
                    if valid.any():
                        cad_cost = torch.zeros((num_queries, num_tgt), device=out_cad.device)
                        cad_prob = out_cad[b].softmax(-1)
                        cad_cost[:, valid] = -cad_prob[:, cad_ids[valid]]
                        total = total + self.cost_cad * cad_cost

            c = total.detach().cpu()
            if _linear_sum_assignment is not None:
                row_ind, col_ind = _linear_sum_assignment(c.numpy())
                row = torch.as_tensor(row_ind, dtype=torch.long)
                col = torch.as_tensor(col_ind, dtype=torch.long)
            else:
                row, col = _greedy_assignment(c)
            indices.append((row, col))
        return indices
