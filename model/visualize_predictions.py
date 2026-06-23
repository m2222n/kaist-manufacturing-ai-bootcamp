from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from depth_vq_detector.visualization import save_prediction_visualization


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize depth VQ detector predictions over a scene depth map.")
    parser.add_argument("--scene_npz", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_preds", type=int, default=20)
    parser.add_argument("--include_gt", action="store_true")
    args = parser.parse_args()

    scene_npz = Path(args.scene_npz)
    data = np.load(scene_npz, allow_pickle=True)
    depth = data["depth"].astype(np.float32)
    pred_data = json.load(open(args.predictions, "r", encoding="utf-8"))
    preds = pred_data.get("predictions", pred_data if isinstance(pred_data, list) else [])[: args.max_preds]
    mask_npz = np.load(args.masks, allow_pickle=True)
    masks = mask_npz["masks"].astype(bool)

    save_prediction_visualization(
        depth=depth,
        predictions=preds,
        masks=masks,
        out_path=args.out,
        scene_npz=scene_npz,
        include_gt=args.include_gt,
    )


if __name__ == "__main__":
    main()
