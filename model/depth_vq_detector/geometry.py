from __future__ import annotations

import torch


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert boxes from normalized/absolute cx,cy,w,h to x1,y1,x2,y2."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Generalized IoU from https://giou.stanford.edu/. Boxes are xyxy."""
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area.clamp(min=1e-6)


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Compute normalized xyxy boxes from masks of shape [N,H,W]."""
    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=masks.device)
    n, h, w = masks.shape
    boxes = []
    for mask in masks:
        y, x = torch.where(mask > 0)
        if x.numel() == 0:
            boxes.append(torch.zeros((4,), dtype=torch.float32, device=masks.device))
        else:
            x0, x1 = x.min().float(), x.max().float() + 1.0
            y0, y1 = y.min().float(), y.max().float() + 1.0
            boxes.append(torch.tensor([x0 / w, y0 / h, x1 / w, y1 / h], dtype=torch.float32, device=masks.device))
    return torch.stack(boxes, dim=0)
