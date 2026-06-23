#!/usr/bin/env python3
"""
Build CAD global embedding memory bank from a trained PointNet++ CAD encoder.

Example:
    python build_cad_memory_bank.py \
        --manifest ./pc_dataset/manifest.json \
        --checkpoint ./runs/cad_pointnet2/best.pt \
        --output ./runs/cad_pointnet2/cad_memory_bank.npz \
        --n_points 4096 \
        --save_local
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cad3d_encoder import CADPointCloudContrastiveDataset, CADPointNet2Model, PointNet2Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_points", type=int, default=None, help="Defaults to checkpoint n_points if omitted")
    p.add_argument("--save_local", action="store_true", help="Also save local_xyz and local_tokens")
    p.add_argument("--torch_num_threads", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_num_threads is not None and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args: Dict = ckpt.get("args", {})
    cfg_dict = ckpt["pointnet2_config"]
    config = PointNet2Config(**cfg_dict)

    n_points = args.n_points if args.n_points is not None else train_args.get("n_points", None)
    use_normals = config.input_dim == 6

    dataset = CADPointCloudContrastiveDataset(
        manifest_path=args.manifest,
        n_points=n_points,
        use_normals=use_normals,
        two_views=False,
        augment=False,
        rotation_aug="none",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = CADPointNet2Model(
        config=config,
        num_classes=len(ckpt.get("class_to_idx", dataset.class_to_idx)),
        num_cads=len(ckpt.get("cad_to_idx", dataset.cad_to_idx)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    cad_ids = []
    class_names = []
    cad_indices = []
    class_indices = []
    embeddings = []
    local_xyz_list = []
    local_tokens_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Building memory bank", ncols=100):
            x = batch["features"].to(device, non_blocking=True)
            out = model(x)
            embeddings.append(out["global_embedding"].detach().cpu().numpy())
            cad_ids.extend(list(batch["cad_id"]))
            class_names.extend(list(batch["class_name"]))
            cad_indices.append(batch["cad_idx"].numpy())
            class_indices.append(batch["class_idx"].numpy())
            if args.save_local:
                local_xyz_list.append(out["local_xyz"].detach().cpu().numpy())
                local_tokens_list.append(out["local_tokens"].detach().cpu().numpy())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "cad_ids": np.asarray(cad_ids),
        "class_names": np.asarray(class_names),
        "cad_indices": np.concatenate(cad_indices, axis=0),
        "class_indices": np.concatenate(class_indices, axis=0),
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float32),
    }
    if args.save_local:
        data["local_xyz"] = np.concatenate(local_xyz_list, axis=0).astype(np.float32)
        data["local_tokens"] = np.concatenate(local_tokens_list, axis=0).astype(np.float32)

    np.savez_compressed(output, **data)
    print(f"Saved memory bank to: {output}")
    print(f"embeddings shape: {data['embeddings'].shape}")
    if args.save_local:
        print(f"local_xyz shape: {data['local_xyz'].shape}")
        print(f"local_tokens shape: {data['local_tokens'].shape}")


if __name__ == "__main__":
    main()
