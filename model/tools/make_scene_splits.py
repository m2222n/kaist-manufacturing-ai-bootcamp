from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _scene_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"(\d+)", path.stem)
    return (int(m.group(1)) if m else 10**12, path.name)


def _json_loads_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            value = str(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    if isinstance(value, dict):
        return value
    return {}


def _load_meta(path: Path) -> dict[str, Any]:
    try:
        data = np.load(path, allow_pickle=True)
        if "meta" not in data.files:
            return {}
        return _json_loads_maybe(data["meta"])
    except Exception:
        return {}


def _discover_scene_files(data_root: Path) -> list[Path]:
    npz_dir = data_root / "npz" if (data_root / "npz").exists() else data_root
    files = list(npz_dir.glob("scene_*.npz")) + list(npz_dir.glob("scene*.npz"))
    unique = []
    seen = set()
    for f in sorted(files, key=_scene_sort_key):
        r = f.resolve()
        if r in seen:
            continue
        seen.add(r)
        unique.append(r)
    if not unique:
        raise FileNotFoundError(f"No scene_*.npz files found under {npz_dir}")
    return unique


def _split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Ratios must have a positive sum.")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    if n >= 3:
        if n_val == 0 and val_ratio > 0:
            n_val = 1
            n_train = max(n_train - 1, 1)
        if n_test == 0 and test_ratio > 0:
            n_test = 1
            n_train = max(n_train - 1, 1)
    return n_train, n_val, n_test


def _assign_random(files: list[Path], seed: int, ratios: tuple[float, float, float]) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    items = list(files)
    rng.shuffle(items)
    n_train, n_val, n_test = _split_counts(len(items), *ratios)
    return {
        "train": sorted(items[:n_train], key=_scene_sort_key),
        "val": sorted(items[n_train:n_train + n_val], key=_scene_sort_key),
        "test": sorted(items[n_train + n_val:n_train + n_val + n_test], key=_scene_sort_key),
    }


def _assign_stratified_by_bg(files: list[Path], seed: int, ratios: tuple[float, float, float]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        meta = _load_meta(f)
        bg = str(meta.get("bg_kind", "unknown"))
        groups[bg].append(f)
    out = {"train": [], "val": [], "test": []}
    for bg, group in sorted(groups.items()):
        sub_seed = int(hashlib.md5((bg + str(seed)).encode("utf-8")).hexdigest()[:8], 16)
        split = _assign_random(group, sub_seed, ratios)
        for k in out:
            out[k].extend(split[k])
    for k in out:
        out[k] = sorted(out[k], key=_scene_sort_key)
    return out


def _relative_to_manifest(path: Path, manifest_path: Path) -> str:
    try:
        return str(path.resolve().relative_to(manifest_path.parent.resolve()))
    except ValueError:
        import os
        return os.path.relpath(path.resolve(), manifest_path.parent.resolve())


def _scene_record(path: Path, manifest_path: Path) -> dict[str, str]:
    return {"scene_id": path.stem, "scene_npz": _relative_to_manifest(path, manifest_path)}


def _stats(files: list[Path]) -> dict[str, Any]:
    bg_counts = Counter()
    class_counts = Counter()
    n_instances = 0
    for f in files:
        meta = _load_meta(f)
        bg_counts[str(meta.get("bg_kind", "unknown"))] += 1
        inst = meta.get("instances", {}) if isinstance(meta, dict) else {}
        if isinstance(inst, dict):
            for _, info in inst.items():
                if isinstance(info, dict):
                    cid = info.get("category_id", info.get("label"))
                    if cid is not None:
                        class_counts[str(int(cid))] += 1
                    n_instances += 1
    return {
        "num_scenes": len(files),
        "num_instances_from_meta": int(n_instances),
        "bg_kind_counts": dict(sorted(bg_counts.items())),
        "category_id_counts": dict(sorted(class_counts.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic scene-level train/val/test splits for depth VQ detector.")
    parser.add_argument("--data_root", required=True, help="Dataset root containing npz/scene_*.npz")
    parser.add_argument("--out_dir", required=True, help="Directory to write train.json, val.json, test.json")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify_bg_kind", action="store_true", help="Keep clear_box/white_desk proportions approximately stable.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _discover_scene_files(data_root)
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    split_files = _assign_stratified_by_bg(files, args.seed, ratios) if args.stratify_bg_kind else _assign_random(files, args.seed, ratios)

    all_ids = []
    for fs in split_files.values():
        all_ids.extend([f.stem for f in fs])
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Split leakage detected: some scene_id appears in more than one split.")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "seed": args.seed,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "stratify_bg_kind": bool(args.stratify_bg_kind),
        "total_scenes": len(files),
        "splits": {},
    }

    for split_name in ["train", "val", "test"]:
        out_path = out_dir / f"{split_name}.json"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists. Use --overwrite to replace it.")
        records = [_scene_record(f, out_path) for f in split_files[split_name]]
        payload = {
            "split": split_name,
            "data_root": str(data_root),
            "seed": args.seed,
            "scenes": records,
            "stats": _stats(split_files[split_name]),
        }
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        summary["splits"][split_name] = {
            "manifest": str(out_path),
            "num_scenes": len(records),
            "stats": payload["stats"],
        }
        print(f"{split_name:>5}: {len(records)} scenes -> {out_path}")

    summary_path = out_dir / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
