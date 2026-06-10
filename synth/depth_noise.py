"""합성 depth -> 실제 Blaze-112 ToF처럼 보이게 노이즈 주입 (sim-to-real).

우리 합성 depth는 '완벽하게 매끈'(배경 0%, 노이즈 0). 실제 Blaze ToF는:
  1. Gaussian noise   — 거리 비례 측정 오차 (ToF는 멀수록 노이즈↑)
  2. pixel dropout    — confidence 낮은 픽셀(어두운/경사진 표면)에서 깊이 누락(=구멍, 0)
  3. flying pixel     — 깊이 불연속 경계에서 앞뒤 깊이가 섞여 튀는 값
  4. lateral noise    — 경계 위치가 살짝 흔들림(블러)

참고(6/9 조사): DREDS(PKU), WISDOM, arXiv 2402.16514 ToF noise model.
Blaze-112 실용범위 0.3~1.5m, 640x480, 펄스 ToF 850/940nm NIR.

용법:
  from depth_noise import add_blaze_noise
  noisy = add_blaze_noise(depth_m, rng)        # depth_m: (H,W) float32, meter
hdf5 후처리:
  python depth_noise.py <src_dir> <out_dir>    # 각 hdf5에 'depth_noisy' 추가 저장
"""
import numpy as np


# --- scipy 없이 numpy만으로 필터 구현 (venv를 BlenderProc 전용으로 깨끗하게 유지) ---
def _box_blur(a, k=5):
    """k×k 박스(평균) 블러. 누적합으로 O(N)."""
    pad = k // 2
    ap = np.pad(a, pad, mode="edge")
    cs = np.cumsum(np.cumsum(ap, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
    H, W = a.shape
    s = (cs[k:k+H, k:k+W] - cs[:H, k:k+W] - cs[k:k+H, :W] + cs[:H, :W])
    return (s / (k * k)).astype(np.float32)


def _gauss_blur(a, sigma=0.7):
    """작은 가우시안 ≈ 3×3 가중 박스로 근사 (경계 흔들림용, 정밀도 불필요)."""
    return _box_blur(a, 3)


def _dilate(mask, it=1):
    """이진 마스크 팽창 (4-이웃), numpy 시프트로."""
    m = mask.copy()
    for _ in range(it):
        d = m.copy()
        d[1:, :] |= m[:-1, :]; d[:-1, :] |= m[1:, :]
        d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
        m = d
    return m


def add_blaze_noise(depth, rng,
                    axial_sigma_base=0.002,   # 2mm 기본 축방향 노이즈(@근거리)
                    axial_sigma_dist=0.004,   # 거리 비례 추가 (1m당 4mm)
                    dropout_rate=0.02,        # 전역 무작위 dropout 비율
                    edge_dropout=0.25,        # 경계 픽셀 추가 dropout 확률
                    flying_strength=0.02,     # flying pixel 깊이 섞임 강도(m)
                    edge_thresh=0.003):       # 경계 판정 깊이경사(m/px)
    """합성 depth(meter)에 Blaze ToF 노이즈를 주입. 0 = 무효(구멍)."""
    d = depth.astype(np.float32).copy()
    valid = d > 0

    # --- 경계(깊이 불연속) 검출: flying pixel / edge dropout이 일어나는 곳 ---
    gy, gx = np.gradient(d)
    grad = np.sqrt(gx**2 + gy**2)
    edges = (grad > edge_thresh) & valid

    # --- 1. 축방향 Gaussian noise (거리 비례) ---
    sigma = axial_sigma_base + axial_sigma_dist * d
    d[valid] += rng.normal(0, 1, d.shape)[valid] * sigma[valid]

    # --- 3. flying pixel: 경계에서 이웃 깊이와 섞여 튐 ---
    if edges.any():
        blurred = _box_blur(d, 5)
        mix = rng.uniform(-flying_strength, flying_strength, d.shape)
        d[edges] = blurred[edges] + mix[edges]

    # --- 4. lateral noise: 경계 위치 흔들림(약한 블러를 경계에만) ---
    sm = _gauss_blur(d, sigma=0.7)
    edge_band = _dilate(edges, it=1)
    d[edge_band] = sm[edge_band]

    # --- 2. pixel dropout: confidence 낮은 픽셀 = 구멍(0) ---
    drop = rng.random(d.shape) < dropout_rate
    drop |= edges & (rng.random(d.shape) < edge_dropout)  # 경계에 dropout 집중
    d[drop] = 0.0
    d[~valid] = 0.0  # 원래 무효였던 곳 유지

    return d.astype(np.float32)


def _process_dir(src_dir, out_dir):
    import h5py, glob, os, shutil
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(777)
    files = sorted(glob.glob(f"{src_dir}/*.hdf5"),
                   key=lambda p: int(os.path.basename(p).split(".")[0]))
    for i, p in enumerate(files):
        name = os.path.basename(p)
        dst = f"{out_dir}/{name}"
        shutil.copy(p, dst)
        with h5py.File(dst, "a") as f:
            d = np.array(f["depth"]).astype(np.float32)
            noisy = add_blaze_noise(d, rng)
            if "depth_noisy" in f:
                del f["depth_noisy"]
            f.create_dataset("depth_noisy", data=noisy, compression="gzip")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} done")
    print(f"완료: {len(files)}개 -> {out_dir}/ (depth_noisy 추가)")


if __name__ == "__main__":
    import sys
    _process_dir(sys.argv[1], sys.argv[2])
