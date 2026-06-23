from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest for provided npz/scene_*.npz dataset")
    parser.add_argument("--root", required=True, help="Dataset root containing npz/, crops/, vis/ or directly scene_*.npz")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    npz_dir = root / "npz" if (root / "npz").exists() else root
    files = sorted(npz_dir.glob("scene_*.npz")) + sorted(npz_dir.glob("scene*.npz"))
    seen = set()
    scenes = []
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        scenes.append({"scene_id": f.stem, "scene_npz": str(f.relative_to(root.parent))})
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump({"scenes": scenes}, fp, indent=2)
    print(f"Wrote {len(scenes)} scenes to {output}")


if __name__ == "__main__":
    main()
