"""내일 미팅 발표용 이미지를 의미별 4개 폴더로 정리.
  1_v1_단색배경/      v1 대표 4패널
  2_v2_도메인랜덤화/   v2 대표 4패널
  3_비교_단색vs랜덤화/  v1 RGB | v2 RGB 나란히
  4_depth_활용/        clean vs Blaze노이즈, depth-only 실험
"""
import h5py, numpy as np, os, sys
from PIL import Image
sys.path.insert(0, "/home/jtm/kaist_project/synth")
from depth_noise import add_blaze_noise

V1 = "/data/jtm/synth_out/dataset_v1"
V2 = "/data/jtm/synth_out/dataset_v2"
BASE = "/data/jtm/synth_out/meeting_0611"

_rng = np.random.default_rng(12345)
PAL = (_rng.random((30, 3)) * 255).astype(np.uint8); PAL[0] = [25, 25, 25]
def col(l): return PAL[np.clip(l, 0, 29)]

def d2gray(d):
    out = np.zeros((*d.shape, 3), np.uint8); v = d[d > 0]
    if v.size:
        lo, hi = v.min(), v.max(); n = np.clip((d - lo) / (hi - lo + 1e-9), 0, 1)
        g = (255 * (1 - n)).astype(np.uint8)
        for c in range(3): out[..., c] = g
    out[d <= 0] = [200, 40, 40]   # 구멍=빨강
    return out

def load(path):
    with h5py.File(path, "r") as f:
        return (np.array(f["colors"])[..., :3].astype(np.uint8),
                np.array(f["depth"]).astype(np.float32),
                np.array(f["instance_segmaps"]).astype(np.int64),
                np.array(f["category_id_segmaps"]).astype(np.int64))

def hcat(imgs, gap=10):
    h = imgs[0].shape[0]; w = imgs[0].shape[1]
    g = np.zeros((h, w * len(imgs) + gap * (len(imgs) - 1), 3), np.uint8)
    for i, im in enumerate(imgs): g[:, i * (w + gap):i * (w + gap) + w] = im
    return g

def panel4(rgb, dep, inst, cat):  # RGB|depth|inst|cat
    return hcat([rgb, d2gray(dep), col(inst), col(cat)])

# ---- 1. v1 단색 ----
d1 = f"{BASE}/1_v1_단색배경"; os.makedirs(d1, exist_ok=True)
for s in [160, 0, 3, 352]:
    rgb, dep, inst, cat = load(f"{V1}/{s}.hdf5")
    Image.fromarray(panel4(rgb, dep, inst, cat)).save(f"{d1}/v1_scene{s}.png")

# ---- 2. v2 도메인 랜덤화 ----
d2 = f"{BASE}/2_v2_도메인랜덤화"; os.makedirs(d2, exist_ok=True)
for s in [6, 16, 11, 0]:
    rgb, dep, inst, cat = load(f"{V2}/{s}.hdf5")
    Image.fromarray(panel4(rgb, dep, inst, cat)).save(f"{d2}/v2_scene{s}.png")

# ---- 3. 비교: v1 단색 RGB | v2 실사 RGB ----
d3 = f"{BASE}/3_비교_단색vs랜덤화"; os.makedirs(d3, exist_ok=True)
for v1s, v2s in [(0, 0), (3, 11)]:
    r1 = load(f"{V1}/{v1s}.hdf5")[0]; r2 = load(f"{V2}/{v2s}.hdf5")[0]
    Image.fromarray(hcat([r1, r2], gap=14)).save(f"{d3}/비교_단색{v1s}_vs_랜덤화{v2s}.png")

# ---- 4. depth 활용 ----
d4 = f"{BASE}/4_depth_활용"; os.makedirs(d4, exist_ok=True)
rng = np.random.default_rng(5)
# 4a. clean vs Blaze 노이즈
rgb, dep, inst, cat = load(f"{V2}/0.hdf5")
noisy = add_blaze_noise(dep, rng)
Image.fromarray(hcat([d2gray(dep), d2gray(noisy)], gap=14)).save(f"{d4}/depth_clean_vs_Blaze노이즈.png")
# 4b. depth-only 실험 (RGB참고 | depth | cat정답) 3장
for s in [3, 16, 11]:
    rgb, dep, inst, cat = load(f"{V2}/{s}.hdf5")
    Image.fromarray(hcat([rgb, d2gray(dep), col(cat)])).save(f"{d4}/depthonly_scene{s}.png")

print("=== 완료 ===")
for d in [d1, d2, d3, d4]:
    print(f"  {d}: {len([f for f in os.listdir(d) if f.endswith('.png')])}장")
