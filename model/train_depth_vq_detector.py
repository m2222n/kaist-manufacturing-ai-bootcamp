from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from depth_vq_detector import DepthInstanceDataset, build_cad_alias_map, collate_fn
from depth_vq_detector.depth_preprocess import input_channels_for_mode
from depth_vq_detector.losses import SetCriterion
from depth_vq_detector.matcher import HungarianMatcher
from depth_vq_detector.model import DepthVQDetector, load_cad_codebook


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


def get_val_source(args: argparse.Namespace) -> str | None:
    return args.val_data_root or args.val_scene_manifest or args.val_scene_npz


def build_dataset(source: str, args: argparse.Namespace, cad_id_to_index: dict[str, int]) -> DepthInstanceDataset:
    return DepthInstanceDataset(
        source,
        cad_id_to_index=cad_id_to_index,
        input_mode=args.input_mode,
        image_size=args.image_size,
        depth_scale=args.depth_scale,
        label_offset=args.label_offset,
        min_mask_area=args.min_mask_area,
    )


def build_loader(dataset: DepthInstanceDataset, args: argparse.Namespace, train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )


def check_cad_matches(dataset: DepthInstanceDataset, split_name: str, cad_ids: list[str], stage: str) -> None:
    if not cad_ids:
        return
    matched = 0
    total = 0
    n = min(len(dataset), 10)
    for i in range(n):
        _, t = dataset[i]
        total += int(t["cad_ids"].numel())
        matched += int((t["cad_ids"] >= 0).sum().item())
    print(f"CAD memory match check [{split_name}] first {n} scenes: {matched}/{total} instances matched by stl/cad_id aliases")
    if stage in {"vq", "joint"} and total > 0 and matched == 0:
        raise RuntimeError(
            f"CAD memory matching is 0 on split={split_name}. Check cad_ids in cad_memory_bank.npz versus "
            "meta.instances[*].stl in the scene npz. Training would otherwise run with "
            "loss_cad_ce=0 and an untrained CAD VQ head."
        )


def build_criterion(args: argparse.Namespace, cad_codebook: torch.Tensor | None) -> SetCriterion:
    cad_loss_on = args.stage in {"vq", "joint"} and cad_codebook is not None
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
        cost_cad=0.2 if cad_loss_on and args.stage == "joint" else 0.0,
    )
    return SetCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        cad_codebook=cad_codebook,
    )




def load_compatible_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"Initialized from {checkpoint_path}: loaded {len(compatible)} tensors, skipped {len(skipped)} incompatible tensors")
    if skipped:
        print("  skipped examples: " + ", ".join(skipped[:8]))
    if missing:
        print("  missing after init examples: " + ", ".join(list(missing)[:8]))
    if unexpected:
        print("  unexpected after init examples: " + ", ".join(list(unexpected)[:8]))


def _accumulate(loss_sums: dict[str, float], losses: dict[str, torch.Tensor]) -> None:
    for k, v in losses.items():
        loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().cpu().item())


def _average(loss_sums: dict[str, float], count: int) -> dict[str, float]:
    denom = max(count, 1)
    return {k: v / denom for k, v in sorted(loss_sums.items())}


def _format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    pieces = [f"{prefix}_{k}={v:.4f}" for k, v in metrics.items()]
    return " ".join(pieces)


def train_one_epoch(
    model: nn.Module,
    criterion: SetCriterion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
    model.train()
    loss_sums: dict[str, float] = {}
    count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
            losses = criterion(outputs, targets)
            loss = losses["loss_total"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        _accumulate(loss_sums, losses)
        count += 1
        global_step += 1
        if args.print_freq > 0 and global_step % args.print_freq == 0:
            metrics = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
            print(f"epoch={epoch} step={global_step} " + _format_metrics("train", metrics))
    return _average(loss_sums, count), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: SetCriterion,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    loss_sums: dict[str, float] = {}
    count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)
        with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
            losses = criterion(outputs, targets)
        _accumulate(loss_sums, losses)
        count += 1
    return _average(loss_sums, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train depth-only VQ query detector on scene_*.npz format")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data_root", default=None, help="Train dataset root containing npz/, crops/, vis/ folders")
    src.add_argument("--scene_manifest", default=None, help="Train JSON manifest; entries can use scene_npz")
    src.add_argument("--scene_npz", default=None, help="Single train scene npz for debugging")

    val_src = parser.add_mutually_exclusive_group(required=False)
    val_src.add_argument("--val_data_root", default=None, help="Validation dataset root. Prefer --val_scene_manifest for fixed split.")
    val_src.add_argument("--val_scene_manifest", default=None, help="Validation JSON manifest generated by tools/make_scene_splits.py")
    val_src.add_argument("--val_scene_npz", default=None, help="Single validation scene npz for debugging")

    parser.add_argument("--cad_memory", default=None, help="cad_memory_bank.npz with embeddings and cad_ids. Optional for --stage det")
    parser.add_argument("--init_checkpoint", default=None, help="Optional checkpoint to initialize compatible model weights before training")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_classes", type=int, default=27, help="Number of object semantic classes, excluding no-object")
    parser.add_argument("--input_mode", default="zv", choices=["z", "zv", "xyzv", "xyznv"])
    parser.add_argument("--image_size", type=parse_image_size, default=None, help="H,W resize, e.g. 512,512. Keep None for native 512.")
    parser.add_argument("--depth_scale", type=float, default=None, help="Optional multiplier. Provided npz depth is already meters, so normally omit.")
    parser.add_argument("--label_offset", type=int, default=1, help="Subtract from category_id. Provided labels are 1..27, so default=1.")
    parser.add_argument("--min_mask_area", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--backbone_dim", type=int, default=64)
    parser.add_argument("--decoder_layers", type=int, default=6)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--stage", choices=["det", "vq", "joint"], default="joint", help="det disables CAD losses; vq/joint enable them")
    parser.add_argument("--eval_interval", type=int, default=1, help="Run validation every N epochs when validation source is provided")
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--clip_grad_norm", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch_num_threads", type=int, default=None)
    args = parser.parse_args()

    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)

    train_source = args.data_root or args.scene_manifest or args.scene_npz
    val_source = get_val_source(args)
    assert train_source is not None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    cad_codebook = None
    cad_ids: list[str] = []
    cad_id_to_index: dict[str, int] = {}
    if args.cad_memory:
        cad_codebook, cad_ids = load_cad_codebook(args.cad_memory)
        cad_id_to_index = build_cad_alias_map(cad_ids)
    elif args.stage in {"vq", "joint"}:
        raise ValueError("--cad_memory is required for --stage vq or --stage joint. Use --stage det for detector-only training.")

    train_dataset = build_dataset(train_source, args, cad_id_to_index)
    print(f"Loaded train split: {len(train_dataset)} scene(s) from {train_source}")
    check_cad_matches(train_dataset, "train", cad_ids, args.stage)

    val_dataset = None
    if val_source is not None:
        val_dataset = build_dataset(val_source, args, cad_id_to_index)
        print(f"Loaded val split: {len(val_dataset)} scene(s) from {val_source}")
        check_cad_matches(val_dataset, "val", cad_ids, args.stage)
    else:
        print("No validation split provided. best.pt will be selected by train loss. For formal experiments, pass --val_scene_manifest.")

    train_loader = build_loader(train_dataset, args, train=True)
    val_loader = build_loader(val_dataset, args, train=False) if val_dataset is not None else None

    device = torch.device(args.device)
    model = DepthVQDetector(
        in_channels=input_channels_for_mode(args.input_mode),
        num_classes=args.num_classes,
        cad_codebook=cad_codebook,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        backbone_dim=args.backbone_dim,
        decoder_layers=args.decoder_layers,
        nheads=args.nheads,
    ).to(device)

    if args.init_checkpoint:
        load_compatible_checkpoint(model, args.init_checkpoint)

    criterion = build_criterion(args, cad_codebook).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_metric = float("inf")
    best_metric_name = "val_loss_total" if val_loader is not None else "train_loss_total"
    global_step = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
            global_step=global_step,
        )

        val_metrics = None
        should_eval = val_loader is not None and args.eval_interval > 0 and (epoch % args.eval_interval == 0 or epoch == args.epochs)
        if should_eval:
            val_metrics = evaluate(model, criterion, val_loader, device, args)

        train_loss = train_metrics.get("loss_total", float("inf"))
        val_loss = val_metrics.get("loss_total", float("inf")) if val_metrics is not None else None
        selected_metric = val_loss if val_loss is not None else train_loss

        log_line = f"[epoch {epoch}/{args.epochs}] " + _format_metrics("train", train_metrics)
        if val_metrics is not None:
            log_line += " | " + _format_metrics("val", val_metrics)
        print(log_line)

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
            "best_metric_name": best_metric_name,
        }
        history.append(epoch_record)
        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "cad_ids": cad_ids,
            "cad_codebook": cad_codebook,
            "epoch": epoch,
            "global_step": global_step,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_metric_name": best_metric_name,
            "selected_metric": selected_metric,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if selected_metric < best_metric:
            best_metric = float(selected_metric)
            ckpt["best_metric"] = best_metric
            torch.save(ckpt, out_dir / "best.pt")
            print(f"Saved best.pt by {best_metric_name}={best_metric:.6f}")

    print(f"Training done. Best {best_metric_name}: {best_metric:.6f}. Checkpoints saved to {out_dir}")


if __name__ == "__main__":
    main()
