from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from depth_vq_detector import DepthInstanceDataset, collate_fn, build_cad_alias_map
from depth_vq_detector.depth_preprocess import input_channels_for_mode
from depth_vq_detector.matcher import HungarianMatcher
from depth_vq_detector.losses import SetCriterion
from depth_vq_detector.model import DepthVQDetector


def parse_image_size(value: str | None) -> tuple[int, int] | None:
    if value is None or str(value).lower() in {"none", ""}:
        return None
    parts = str(value).lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("image_size must be like 512,512 or 512x512")
    return int(parts[0]), int(parts[1])


def move_targets_to_device(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    out = []
    for t in targets:
        nt = {}
        for k, v in t.items():
            nt[k] = v.to(device) if torch.is_tensor(v) else v
        out.append(nt)
    return out


def choose_source(args: argparse.Namespace) -> str:
    return args.scene_manifest or args.data_root or args.scene_npz


def box_iou_diag(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))
    x0 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y0 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    x1 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y1 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter).clamp(min=1e-6)


def make_criterion(args_like: dict[str, Any], num_classes: int, cad_codebook: torch.Tensor | None) -> tuple[SetCriterion, HungarianMatcher]:
    stage = str(args_like.get("stage", "joint"))
    cad_loss_on = stage in {"vq", "joint"} and cad_codebook is not None
    weight_dict = {
        "loss_ce": 2.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_mask": 2.0,
        "loss_dice": 2.0,
        "loss_cad_ce": 1.0 if cad_loss_on else 0.0,
        "loss_cad_align": 0.2 if cad_loss_on else 0.0,
    }
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        cost_mask=2.0,
        cost_dice=2.0,
        cost_cad=0.2 if cad_loss_on and stage == "joint" else 0.0,
    )
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        cad_codebook=cad_codebook,
    )
    return criterion, matcher


@torch.no_grad()
def matched_metrics(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, Any]],
    matcher: HungarianMatcher,
    mask_thresh: float,
) -> dict[str, float]:
    from depth_vq_detector.geometry import box_cxcywh_to_xyxy

    indices = matcher(outputs, targets)
    device = outputs["pred_logits"].device
    sums = {
        "gt_objects": 0.0,
        "matched": 0.0,
        "class_correct": 0.0,
        "cad_valid": 0.0,
        "cad_correct": 0.0,
        "box_iou_sum": 0.0,
        "mask_iou_sum": 0.0,
    }
    for b, (src_cpu, tgt_cpu) in enumerate(indices):
        sums["gt_objects"] += float(targets[b]["labels"].numel())
        if src_cpu.numel() == 0:
            continue
        src = src_cpu.to(device)
        tgt = tgt_cpu.to(device)
        n = int(src.numel())
        sums["matched"] += n

        pred_labels = outputs["pred_logits"][b, src, :-1].argmax(-1)
        tgt_labels = targets[b]["labels"].to(device)[tgt]
        sums["class_correct"] += float((pred_labels == tgt_labels).sum().item())

        pred_boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"][b, src])
        tgt_boxes = targets[b]["boxes_xyxy"].to(device)[tgt]
        sums["box_iou_sum"] += float(box_iou_diag(pred_boxes, tgt_boxes).sum().item())

        pred_masks = outputs["pred_masks"][b, src]
        tgt_masks = targets[b]["masks"].to(device)[tgt].bool()
        pred_masks = F.interpolate(pred_masks[:, None], size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        pred_masks = pred_masks.sigmoid() > mask_thresh
        inter = (pred_masks & tgt_masks).flatten(1).sum(-1).float()
        union = (pred_masks | tgt_masks).flatten(1).sum(-1).float().clamp(min=1.0)
        sums["mask_iou_sum"] += float((inter / union).sum().item())

        if "pred_cad_logits" in outputs:
            tgt_cad = targets[b]["cad_ids"].to(device)[tgt]
            valid = tgt_cad >= 0
            if valid.any():
                pred_cad = outputs["pred_cad_logits"][b, src].argmax(-1)
                sums["cad_valid"] += float(valid.sum().item())
                sums["cad_correct"] += float((pred_cad[valid] == tgt_cad[valid]).sum().item())
    return sums


def finalize(loss_sums: dict[str, float], metric_sums: dict[str, float], batches: int) -> dict[str, float]:
    out = {k: v / max(batches, 1) for k, v in sorted(loss_sums.items())}
    matched = max(metric_sums.get("matched", 0.0), 1.0)
    gt = max(metric_sums.get("gt_objects", 0.0), 1.0)
    cad_valid = metric_sums.get("cad_valid", 0.0)
    out.update({
        "num_gt_objects": metric_sums.get("gt_objects", 0.0),
        "num_matched": metric_sums.get("matched", 0.0),
        "hungarian_recall": metric_sums.get("matched", 0.0) / gt,
        "matched_class_acc": metric_sums.get("class_correct", 0.0) / matched,
        "matched_box_iou": metric_sums.get("box_iou_sum", 0.0) / matched,
        "matched_mask_iou": metric_sums.get("mask_iou_sum", 0.0) / matched,
        "num_cad_valid": cad_valid,
        "matched_cad_acc": metric_sums.get("cad_correct", 0.0) / cad_valid if cad_valid > 0 else None,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate depth-only VQ detector on val/test split")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene_manifest", default=None, help="Split manifest, e.g. splits/val.json or splits/test.json")
    src.add_argument("--data_root", default=None, help="Dataset root containing npz/")
    src.add_argument("--scene_npz", default=None, help="Single scene npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--input_mode", default=None, choices=["z", "zv", "xyzv", "xyznv"])
    parser.add_argument("--image_size", type=parse_image_size, default=None)
    parser.add_argument("--depth_scale", type=float, default=None)
    parser.add_argument("--label_offset", type=int, default=None)
    parser.add_argument("--min_mask_area", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    source = choose_source(args)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    input_mode = args.input_mode or ckpt_args.get("input_mode", "zv")
    image_size = args.image_size or ckpt_args.get("image_size", None)
    if isinstance(image_size, list):
        image_size = tuple(image_size)
    label_offset = int(args.label_offset if args.label_offset is not None else ckpt_args.get("label_offset", 1))
    num_classes = int(ckpt_args.get("num_classes", 27))

    cad_codebook = ckpt.get("cad_codebook")
    cad_ids = [str(x) for x in ckpt.get("cad_ids", [])]
    cad_id_to_index = build_cad_alias_map(cad_ids) if cad_ids else {}

    dataset = DepthInstanceDataset(
        source,
        cad_id_to_index=cad_id_to_index,
        input_mode=input_mode,
        image_size=image_size,
        depth_scale=args.depth_scale if args.depth_scale is not None else ckpt_args.get("depth_scale", None),
        label_offset=label_offset,
        min_mask_area=args.min_mask_area,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    model = DepthVQDetector(
        in_channels=input_channels_for_mode(input_mode),
        num_classes=num_classes,
        cad_codebook=cad_codebook,
        num_queries=int(ckpt_args.get("num_queries", 100)),
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        backbone_dim=int(ckpt_args.get("backbone_dim", 64)),
        decoder_layers=int(ckpt_args.get("decoder_layers", 6)),
        nheads=int(ckpt_args.get("nheads", 8)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    criterion, matcher = make_criterion(ckpt_args, num_classes, cad_codebook)
    criterion.to(device)
    matcher.to(device)

    loss_sums: dict[str, float] = {}
    metric_sums: dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = move_targets_to_device(targets, device)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                losses = criterion(outputs, targets)
            for k, v in losses.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().cpu().item())
            m = matched_metrics(outputs, targets, matcher, args.mask_thresh)
            for k, v in m.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            batches += 1

    metrics = finalize(loss_sums, metric_sums, batches)
    metrics["source"] = str(source)
    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["num_scenes"] = len(dataset)

    out_json = Path(args.out_json) if args.out_json else Path(args.checkpoint).resolve().parent / f"eval_{Path(source).stem}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {out_json}")


if __name__ == "__main__":
    main()
