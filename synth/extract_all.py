"""dataset_v1 전체(2000장)를 4패널 PNG로 추출. 미팅 시연용.
각 PNG = RGB | depth | instance-seg | category-seg
사용: python extract_all.py
"""
import os, glob, json
import h5py
import numpy as np
from PIL import Image

import sys as _sys
_args = _sys.argv[1:]
# --depth-key 로 4패널 depth 자리에 쓸 키 지정 (v2_noisy면 depth_noisy)
DEPTH_KEY = "depth"
if "--depth-key" in _args:
    _i = _args.index("--depth-key")
    DEPTH_KEY = _args[_i + 1]
    _args = _args[:_i] + _args[_i + 2:]
DATASET = _args[0] if len(_args) > 0 else "/data/jtm/synth_out/dataset_v1"
OUT = _args[1] if len(_args) > 1 else "/data/jtm/synth_out/preview_all"
os.makedirs(OUT, exist_ok=True)

_rng = np.random.default_rng(12345)
_PALETTE = (_rng.random((30, 3)) * 255).astype(np.uint8)
_PALETTE[0] = [30, 30, 30]  # 배경 어둡게

def colorize(label):
    return _PALETTE[np.clip(label, 0, 29)]

def depth_to_rgb(depth):
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    lo, hi = valid.min(), valid.max()
    norm = np.clip((depth - lo) / (hi - lo + 1e-9), 0, 1)
    norm[depth <= 0] = 0
    g = (255 * (1 - norm)).astype(np.uint8)  # 가까움=밝음
    return np.stack([g, g, g], axis=-1)

def panel(path, out):
    with h5py.File(path, "r") as f:
        rgb = np.array(f["colors"])[..., :3].astype(np.uint8)
        dkey = DEPTH_KEY if DEPTH_KEY in f else "depth"
        depth = np.array(f[dkey]).astype(np.float32)
        inst = np.array(f["instance_segmaps"]).astype(np.int64)
        cat = np.array(f["category_id_segmaps"]).astype(np.int64)
    imgs = [rgb, depth_to_rgb(depth), colorize(inst), colorize(cat)]
    h, w = rgb.shape[:2]
    grid = np.zeros((h, w * 4 + 30, 3), np.uint8)
    for i, img in enumerate(imgs):
        grid[:, i * (w + 10):i * (w + 10) + w] = img
    Image.fromarray(grid).save(out)

files = sorted(glob.glob(f"{DATASET}/*.hdf5"),
               key=lambda p: int(os.path.basename(p).split(".")[0]))
n = len(files)
for i, p in enumerate(files):
    scene = os.path.basename(p).split(".")[0]
    panel(p, f"{OUT}/scene{int(scene):04d}.png")
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{n} done")
print(f"\n완료: {n}장 -> {OUT}/")
