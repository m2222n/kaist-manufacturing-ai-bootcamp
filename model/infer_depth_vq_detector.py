from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from depth_vq_detector.depth_preprocess import load_depth, load_json, make_depth_input, input_channels_for_mode, resize_depth_nan_safe
from depth_vq_detector.model import DepthVQDetector
from depth_vq_detector.postprocess import postprocess_outputs, prediction_masks_np
from depth_vq_detector.visualization import save_prediction_visualization


def parse_image_size(value: str | None):
    if value is None or str(value).lower() in {"none", ""}:
        return None
    parts = str(value).lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("image_size must be like 512,512 or 512x512")
    return int(parts[0]), int(parts[1])


def _load_scene_npz(path: str | Path) -> tuple[np.ndarray, dict[str, Any], str]:
    data = np.load(path, allow_pickle=True)
    if "depth" not in data.files:
        raise KeyError(f"{path} has no key 'depth'. Available: {data.files}")
    depth = data["depth"].astype(np.float32)
    meta = {}
    if "meta" in data.files:
        raw = data["meta"].item() if getattr(data["meta"], "shape", None) == () else str(data["meta"])
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            try:
                meta = json.loads(raw)
            except json.JSONDecodeError:
                meta = {}
    scene_id = f"scene_{int(meta['scene_idx']):05d}" if "scene_idx" in meta else Path(path).stem
    camera = meta.get("camera", {}) if isinstance(meta.get("camera", {}), dict) else {}
    return depth, camera, scene_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer depth-only VQ query detector")
    parser.add_argument("--checkpoint", required=True)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene_npz", default=None, help="Provided scene npz with key depth")
    src.add_argument("--depth", default=None, help="Standalone depth .npy/.npz/.png")
    parser.add_argument("--camera", default=None, help="Optional camera.json for standalone depth")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--input_mode", default=None, choices=["z", "zv", "xyzv", "xyznv"])
    parser.add_argument("--image_size", type=parse_image_size, default=None)
    parser.add_argument("--depth_scale", type=float, default=None)
    parser.add_argument("--score_thresh", type=float, default=0.25)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--score_mode", choices=["det", "product", "cad"], default="det", help="Score used for filtering/ranking. det=class/object score; product=class*CAD; cad=CAD only.")
    parser.add_argument("--debug_scores", action="store_true", help="Print top query class/CAD scores before thresholding")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true", help="Save overlay PNG with predicted masks, boxes, class ids, and CAD ids")
    parser.add_argument("--include_gt_visualization", action="store_true", help="If --scene_npz is used, add a GT overlay panel to the visualization")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    input_mode = args.input_mode or ckpt_args.get("input_mode", "zv")
    image_size = args.image_size or ckpt_args.get("image_size", None)
    if isinstance(image_size, list):
        image_size = tuple(image_size)
    label_offset = int(ckpt_args.get("label_offset", 1))

    cad_codebook = ckpt.get("cad_codebook")
    cad_ids = [str(x) for x in ckpt.get("cad_ids", [])]
    if cad_codebook is None:
        state = ckpt["model"]
        cad_codebook = state.get("cad_codebook", None)
    # Detector-only checkpoints may have no codebook.

    num_classes = int(ckpt_args.get("num_classes", 27))
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

    if args.scene_npz:
        depth, camera, scene_id = _load_scene_npz(args.scene_npz)
        if args.depth_scale is not None:
            depth = depth * float(args.depth_scale)
        viz_depth = depth.copy()
    else:
        camera = load_json(args.camera)
        depth_scale = args.depth_scale if args.depth_scale is not None else ckpt_args.get("depth_scale", None)
        depth = load_depth(args.depth, depth_scale=depth_scale)
        scene_id = Path(args.depth).stem
        viz_depth = depth.copy()

    depth = resize_depth_nan_safe(depth, image_size)
    x = make_depth_input(depth, camera=camera, mode=input_mode)
    inp = torch.from_numpy(x)[None].float().to(device)

    with torch.no_grad():
        outputs = model(inp)
    if args.debug_scores:
        probs = outputs["pred_logits"][0].softmax(-1)
        class_probs, labels = probs[:, :-1].max(-1)
        print(f"max class_score={float(class_probs.max()):.6f}, mean class_score={float(class_probs.mean()):.6f}")
        if "pred_cad_logits" in outputs:
            cad_prob = outputs["pred_cad_logits"][0].softmax(-1)
            cad_scores, cad_labels = cad_prob.max(-1)
            combined = class_probs * cad_scores
            print(f"max cad_score={float(cad_scores.max()):.6f}, mean cad_score={float(cad_scores.mean()):.6f}")
            print(f"max class*cad={float(combined.max()):.6f}")
        vals, idxs = torch.topk(class_probs, k=min(10, class_probs.numel()))
        for rank, (v, q) in enumerate(zip(vals.tolist(), idxs.tolist()), 1):
            msg = f"top{rank}: q={q} class_id={int(labels[q]) + label_offset} class_score={v:.6f}"
            if "pred_cad_logits" in outputs:
                msg += f" cad_score={float(cad_scores[q]):.6f} cad_index={int(cad_labels[q])}"
            print(msg)
    h, w = depth.shape
    preds = postprocess_outputs(
        outputs,
        image_size=(h, w),
        cad_ids=cad_ids,
        score_thresh=args.score_thresh,
        topk=args.topk,
        mask_thresh=args.mask_thresh,
        class_id_offset=label_offset,
        score_mode=args.score_mode,
    )
    query_indices = [p["query_index"] for p in preds]
    masks = prediction_masks_np(outputs, (h, w), query_indices, mask_thresh=args.mask_thresh)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.json").open("w", encoding="utf-8") as f:
        json.dump({"scene_id": scene_id, "predictions": preds}, f, indent=2)
    np.savez_compressed(out_dir / "predicted_masks.npz", masks=masks.astype(np.uint8), query_indices=np.array(query_indices, dtype=np.int64))
    if args.visualize:
        save_prediction_visualization(
            depth=viz_depth if args.scene_npz else depth,
            predictions=preds,
            masks=masks,
            out_path=out_dir / "visualization.png",
            scene_npz=args.scene_npz,
            include_gt=bool(args.include_gt_visualization and args.scene_npz),
        )
    print(f"Saved {len(preds)} predictions to {out_dir}")


if __name__ == "__main__":
    main()
