#!/usr/bin/env python3
"""
Train a PointNet++ CAD encoder on point clouds generated from STL files.

Example:
    python train_cad_encoder.py \
        --manifest ./pc_dataset/manifest.json \
        --out_dir ./runs/cad_pointnet2 \
        --epochs 100 \
        --batch_size 16 \
        --n_points 4096 \
        --rotation_aug none
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from cad3d_encoder import CADPointCloudContrastiveDataset, CADPointNet2Model, PointNet2Config
from cad3d_encoder.losses import supervised_contrastive_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True, help="Path to point-cloud dataset manifest.json")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to save checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_points", type=int, default=4096)
    p.add_argument("--use_normals", action="store_true", default=True)
    p.add_argument("--no_normals", action="store_false", dest="use_normals")
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--local_dim", type=int, default=256)
    p.add_argument("--sa1_npoint", type=int, default=1024)
    p.add_argument("--sa2_npoint", type=int, default=256)
    p.add_argument("--sa1_radius", type=float, default=0.08)
    p.add_argument("--sa2_radius", type=float, default=0.16)
    p.add_argument("--sa1_nsample", type=int, default=32)
    p.add_argument("--sa2_nsample", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--contrastive_weight", type=float, default=1.0)
    p.add_argument("--cad_id_weight", type=float, default=1.0)
    p.add_argument("--class_weight", type=float, default=0.2)
    p.add_argument("--rotation_aug", type=str, default="none", choices=["none", "z", "so3"])
    p.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--torch_num_threads", type=int, default=1, help="CPU thread count; 1 is often fastest for pure-PyTorch FPS on small/medium batches")
    return p.parse_args()


def make_checkpoint(
    model: CADPointNet2Model,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    dataset: CADPointCloudContrastiveDataset,
    metrics: Dict[str, float],
) -> Dict:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
        "class_to_idx": dataset.class_to_idx,
        "cad_to_idx": dataset.cad_to_idx,
        "pointnet2_config": vars(model.encoder.config),
    }


def main() -> None:
    args = parse_args()
    if args.torch_num_threads is not None and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 6 if args.use_normals else 3

    dataset = CADPointCloudContrastiveDataset(
        manifest_path=args.manifest,
        n_points=args.n_points,
        use_normals=args.use_normals,
        two_views=True,
        augment=True,
        rotation_aug=args.rotation_aug,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=len(dataset) >= args.batch_size,
        pin_memory=torch.cuda.is_available(),
    )

    config = PointNet2Config(
        input_dim=input_dim,
        embedding_dim=args.embedding_dim,
        local_dim=args.local_dim,
        sa1_npoint=args.sa1_npoint,
        sa1_radius=args.sa1_radius,
        sa1_nsample=args.sa1_nsample,
        sa2_npoint=args.sa2_npoint,
        sa2_radius=args.sa2_radius,
        sa2_nsample=args.sa2_nsample,
        dropout=args.dropout,
    )
    model = CADPointNet2Model(
        config=config,
        num_classes=len(dataset.class_to_idx),
        num_cads=len(dataset.cad_to_idx),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "class_to_idx": dataset.class_to_idx,
                "cad_to_idx": dataset.cad_to_idx,
                "pointnet2_config": vars(config),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Device: {device}")
    print(f"Dataset items: {len(dataset)} | classes: {len(dataset.class_to_idx)} | CAD IDs: {len(dataset.cad_to_idx)}")
    print(f"Input dim: {input_dim} | rotation_aug: {args.rotation_aug}")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = {
            "loss": 0.0,
            "loss_contrastive": 0.0,
            "loss_cad": 0.0,
            "loss_class": 0.0,
            "cad_acc": 0.0,
            "class_acc": 0.0,
        }
        n_steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)
        for batch in pbar:
            x1 = batch["features"].to(device, non_blocking=True)
            x2 = batch["features_view2"].to(device, non_blocking=True)
            cad_idx = batch["cad_idx"].to(device, non_blocking=True)
            class_idx = batch["class_idx"].to(device, non_blocking=True)

            x = torch.cat([x1, x2], dim=0)
            cad_labels = torch.cat([cad_idx, cad_idx], dim=0)
            class_labels = torch.cat([class_idx, class_idx], dim=0)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp and device.type == "cuda"):
                out = model(x)
                z = out["global_embedding"]
                loss_con = supervised_contrastive_loss(z, cad_labels, temperature=args.temperature)
                loss_cad = F.cross_entropy(out["cad_logits"], cad_labels)
                loss_class = F.cross_entropy(out["class_logits"], class_labels)
                loss = (
                    args.contrastive_weight * loss_con
                    + args.cad_id_weight * loss_cad
                    + args.class_weight * loss_class
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                cad_acc = (out["cad_logits"].argmax(dim=-1) == cad_labels).float().mean()
                class_acc = (out["class_logits"].argmax(dim=-1) == class_labels).float().mean()

            values = {
                "loss": float(loss.detach().cpu()),
                "loss_contrastive": float(loss_con.detach().cpu()),
                "loss_cad": float(loss_cad.detach().cpu()),
                "loss_class": float(loss_class.detach().cpu()),
                "cad_acc": float(cad_acc.detach().cpu()),
                "class_acc": float(class_acc.detach().cpu()),
            }
            for k, v in values.items():
                running[k] += v
            n_steps += 1

            pbar.set_postfix({k: f"{running[k] / n_steps:.4f}" for k in ["loss", "cad_acc", "class_acc"]})

        metrics = {k: v / max(n_steps, 1) for k, v in running.items()}
        metrics["epoch_time_sec"] = time.time() - t0
        print("Epoch metrics:", json.dumps(metrics, indent=2))

        ckpt = make_checkpoint(model, optimizer, epoch, args, dataset, metrics)
        torch.save(ckpt, out_dir / "latest.pt")
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(ckpt, out_dir / "best.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, out_dir / f"epoch_{epoch:04d}.pt")

    print(f"Done. Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    # PyTorch dataloader on Windows needs this entrypoint guard.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
