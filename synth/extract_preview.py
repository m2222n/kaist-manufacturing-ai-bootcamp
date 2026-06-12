"""hdf5 -> 4패널 PNG (RGB / depth / instance seg / category seg) 육안 검증용.
사용: python extract_preview.py 160 0 3 352
     python extract_preview.py --dataset /data/jtm/synth_out/test_v3 --out /data/jtm/synth_out/preview_v3 0 1
"""
import sys, json
import h5py
import numpy as np
from PIL import Image

DATASET = "/data/jtm/synth_out/dataset_v1"
OUT = "/data/jtm/synth_out/preview"

# --dataset / --out 옵션으로 경로 오버라이드 (없으면 위 기본값)
_args = sys.argv[1:]
def _take(flag, default):
    global _args
    if flag in _args:
        i = _args.index(flag)
        val = _args[i + 1]
        _args = _args[:i] + _args[i + 2:]
        return val
    return default
DATASET = _take("--dataset", DATASET)
OUT = _take("--out", OUT)

import os
os.makedirs(OUT, exist_ok=True)

# 구분 잘 되는 컬러맵 (instance/category 라벨용)
def colorize(label, max_n=27):
    rng = np.random.default_rng(12345)
    palette = (rng.random((max_n + 1, 3)) * 255).astype(np.uint8)
    palette[0] = [30, 30, 30]  # 배경 = 어둡게
    return palette[np.clip(label, 0, max_n)]

def depth_to_rgb(depth):
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = valid.min(), valid.max()
    norm = np.clip((depth - lo) / (hi - lo + 1e-9), 0, 1)
    norm[depth <= 0] = 0
    # 가까움=밝음 반전 (turbo 비슷한 단순 매핑)
    g = (255 * (1 - norm)).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)

def panel(scene):
    p = f"{DATASET}/{scene}.hdf5"
    with h5py.File(p, "r") as f:
        rgb = np.array(f["colors"])[..., :3].astype(np.uint8)
        depth = np.array(f["depth"]).astype(np.float32)
        inst = np.array(f["instance_segmaps"]).astype(np.int64)
        cat = np.array(f["category_id_segmaps"]).astype(np.int64)
        attr = json.loads(bytes(np.array(f["instance_attribute_maps"])).decode())

    n_inst = len(np.unique(inst)) - (1 if 0 in inst else 0)
    cats = sorted(set(a["category_id"] for a in attr if a["category_id"] > 0))
    d = depth_to_rgb(depth)
    insc = colorize(inst)
    catc = colorize(cat)

    h, w = rgb.shape[:2]
    grid = np.zeros((h, w * 4 + 30, 3), np.uint8)
    for i, img in enumerate([rgb, d, insc, catc]):
        grid[:, i * (w + 10):i * (w + 10) + w] = img
    Image.fromarray(grid).save(f"{OUT}/scene{scene}_4panel.png")
    print(f"scene {scene}: instances={n_inst}  categories={cats}  "
          f"depth[{depth[depth>0].min():.3f}~{depth.max():.3f}m]  -> scene{scene}_4panel.png")

for s in _args:
    panel(int(s))
print(f"\n저장 위치: {OUT}/  (RGB | depth | instance-seg | category-seg)")
