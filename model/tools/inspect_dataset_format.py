from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scalar(x):
    if isinstance(x, np.ndarray) and x.shape == ():
        return x.item()
    return x


def inspect_scene(path: Path) -> None:
    d = np.load(path, allow_pickle=True)
    print(f"SCENE {path}")
    print("  keys:", d.files)
    for k in ["depth", "inst_id", "category_id", "meta"]:
        if k not in d.files:
            print(f"  missing {k}")
            continue
        arr = d[k]
        print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")
        if k == "depth":
            print(f"    finite={np.isfinite(arr).sum()} nan={np.isnan(arr).sum()} min={np.nanmin(arr):.6f} max={np.nanmax(arr):.6f}")
        elif k in {"inst_id", "category_id"}:
            vals = np.unique(arr)
            print(f"    unique_count={len(vals)} first={vals[:20]}")
        elif k == "meta":
            raw = scalar(arr)
            try:
                meta = json.loads(raw)
                print("    meta keys:", sorted(meta.keys()))
                print("    visible_inst_ids:", meta.get("visible_inst_ids"))
                first = next(iter(meta.get("instances", {}).items()), None)
                print("    first instance:", first)
            except Exception as e:
                print("    meta parse failed:", e)


def inspect_crop(path: Path) -> None:
    d = np.load(path, allow_pickle=True)
    print(f"CROP {path}")
    print("  keys:", d.files)
    for k in d.files:
        arr = d[k]
        print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")
        if k == "depth":
            print(f"    finite={np.isfinite(arr).sum()} nan={np.isnan(arr).sum()} min={np.nanmin(arr):.6f} max={np.nanmax(arr):.6f}")
        elif k == "mask":
            print(f"    true={arr.astype(bool).sum()}")
        elif arr.shape == ():
            print(f"    value={arr.item()}")
        elif arr.ndim == 1:
            print(f"    value={arr.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect provided scene/crop npz files")
    parser.add_argument("--root", required=True)
    parser.add_argument("--max", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.root)
    scene_files = sorted((root / "npz").glob("*.npz")) if (root / "npz").exists() else sorted(root.glob("scene_*.npz"))
    crop_files = sorted((root / "crops").glob("*.npz")) if (root / "crops").exists() else []
    for p in scene_files[: args.max]:
        inspect_scene(p)
    for p in crop_files[: args.max]:
        inspect_crop(p)


if __name__ == "__main__":
    main()
