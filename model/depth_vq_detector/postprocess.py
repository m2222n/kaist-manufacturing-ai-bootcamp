from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .geometry import box_cxcywh_to_xyxy


@torch.no_grad()
def postprocess_outputs(
    outputs: dict[str, torch.Tensor],
    image_size: tuple[int, int],
    cad_ids: list[str] | None = None,
    score_thresh: float = 0.25,
    topk: int = 100,
    mask_thresh: float = 0.5,
    class_id_offset: int = 0,
    score_mode: str = "det",
) -> list[dict[str, Any]]:
    """Post-process a single image output into Python dict predictions."""
    h, w = image_size
    pred_logits = outputs["pred_logits"][0]
    pred_boxes = outputs["pred_boxes"][0]
    pred_masks = outputs["pred_masks"][0]
    probs = pred_logits.softmax(-1)
    class_probs, labels = probs[:, :-1].max(-1)  # exclude no-object

    cad_labels = None
    cad_scores = None
    if "pred_cad_logits" in outputs:
        cad_prob = outputs["pred_cad_logits"][0].softmax(-1)
        cad_scores, cad_labels = cad_prob.max(-1)
        if score_mode == "product":
            scores = class_probs * cad_scores
        elif score_mode == "cad":
            scores = cad_scores
        elif score_mode == "det":
            scores = class_probs
        else:
            raise ValueError(f"Unknown score_mode={score_mode!r}; expected det, product, or cad")
    else:
        scores = class_probs

    keep = scores >= score_thresh
    if keep.sum() == 0:
        return []
    keep_idx = torch.where(keep)[0]
    keep_scores = scores[keep_idx]
    order = torch.argsort(keep_scores, descending=True)[:topk]
    keep_idx = keep_idx[order]

    masks = F.interpolate(pred_masks[keep_idx, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    masks_prob = masks.sigmoid()
    boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[keep_idx]).clamp(0, 1)
    boxes_abs = boxes_xyxy.clone()
    boxes_abs[:, [0, 2]] *= w
    boxes_abs[:, [1, 3]] *= h

    results = []
    for n, q_idx in enumerate(keep_idx.tolist()):
        cad_index = int(cad_labels[q_idx].item()) if cad_labels is not None else -1
        result = {
            "query_index": int(q_idx),
            "score": float(scores[q_idx].item()),
            "class_id": int(labels[q_idx].item()) + int(class_id_offset),
            "class_score": float(class_probs[q_idx].item()),
            "bbox_xyxy": [float(v) for v in boxes_abs[n].tolist()],
            "mask_area": int((masks_prob[n] > mask_thresh).sum().item()),
        }
        if cad_labels is not None:
            result["cad_index"] = cad_index
            result["cad_score"] = float(cad_scores[q_idx].item())
            result["cad_id"] = cad_ids[cad_index] if cad_ids is not None and 0 <= cad_index < len(cad_ids) else str(cad_index)
        results.append(result)
    return results


@torch.no_grad()
def prediction_masks_np(
    outputs: dict[str, torch.Tensor],
    image_size: tuple[int, int],
    query_indices: list[int],
    mask_thresh: float = 0.5,
):
    """Return boolean masks [N,H,W] for selected query indices."""
    if not query_indices:
        import numpy as np
        h, w = image_size
        return np.zeros((0, h, w), dtype=bool)
    pred_masks = outputs["pred_masks"][0, query_indices]
    masks = F.interpolate(pred_masks[:, None], size=image_size, mode="bilinear", align_corners=False)[:, 0]
    return (masks.sigmoid() > mask_thresh).cpu().numpy()
