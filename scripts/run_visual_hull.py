#!/usr/bin/env python
"""
한 부품에 대해 Visual Hull 복원 → 원본 STL 과 정량 비교.

사용:
  ./venv/bin/python scripts/run_visual_hull.py \
      --views-dir ~/kaist_render/KAIST_dataset_v1/02_sol_block_b \
      --stl ~/kaist_render/stl/02_sol_block_b.stl \
      --res 96 --out outputs/02_sol_block_b

입력: 한 부품 폴더의 다각도 PNG (파일명 _y{tilt}_z{spin}.png) + 원본 STL.
출력: 복원 mesh(.ply) + 비교 지표(JSON, stdout).

⚠️ 정합 전제: 이미지는 render_8views.py 로 생성됐고, 카메라 거리는
   동일 공식 cam_dist = (bbox 대각/2)/tan(30도)*fit 로 STL 에서 재계산한다.
"""
import os
import sys
import glob
import json
import math
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import camera as cam
from src import silhouette as sil
from src import visual_hull as vh
from src import metrics as met


def compute_cam_dist(mesh, fit=1.05, yfov_deg=cam.YFOV_DEG):
    """render_8views.render_part 와 동일한 카메라 거리 공식."""
    radius = float(np.linalg.norm(mesh.bounding_box.extents) / 2.0)
    if radius <= 0:
        radius = 1.0
    return radius / math.tan(math.radians(yfov_deg / 2.0)) * fit


def load_views(views_dir, bg_thresh):
    """폴더의 모든 PNG → [(tilt, spin, mask), ...] + 이미지 크기."""
    paths = sorted(glob.glob(os.path.join(views_dir, "*.png")))
    if not paths:
        raise SystemExit(f"PNG 없음: {views_dir}")
    views = []
    size = None
    for p in paths:
        tilt, spin = cam.parse_angles(os.path.basename(p))
        mask = sil.load_silhouette(p, bg_thresh=bg_thresh)
        if size is None:
            size = mask.shape[0]
        views.append((tilt, spin, mask))
    return views, size, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views-dir", required=True, help="한 부품 다각도 PNG 폴더")
    ap.add_argument("--stl", required=True, help="원본 STL (GT)")
    ap.add_argument("--res", type=int, default=96, help="voxel grid 해상도(축당 개수)")
    ap.add_argument("--bg-thresh", type=int, default=245, help="배경 흰색 임계값")
    ap.add_argument("--out", default=None, help="복원 mesh 출력 prefix (없으면 저장 안 함)")
    ap.add_argument("--iou-pitch", type=float, default=None,
                    help="IoU voxelize pitch(월드 단위). 기본 = bbox/64")
    args = ap.parse_args()

    import trimesh

    # 1) GT mesh 로드 + 정규화 (렌더와 동일하게 원점 중심)
    gt = trimesh.load(os.path.expanduser(args.stl), force="mesh")
    gt.apply_translation(-gt.bounding_box.centroid)
    cam_dist = compute_cam_dist(gt)
    extents = gt.bounding_box.extents
    print(f"[부품] {os.path.basename(args.stl)}  bbox={extents.round(2)}  cam_dist={cam_dist:.2f}")

    # 2) 뷰 + 실루엣
    views, size, paths = load_views(os.path.expanduser(args.views_dir), args.bg_thresh)
    K = cam.intrinsic_matrix(size)
    fg_frac = np.mean([v[2].mean() for v in views])
    print(f"[뷰] {len(views)}장, size={size}, K_f={K[0,0]:.1f}, 평균 전경비율={fg_frac:.3f}")

    # 3) voxel grid (GT bbox 를 살짝 키워서 — Hull 이 bbox 와 같거나 작음)
    pad = 0.10 * extents
    lo = -extents / 2 - pad
    hi = extents / 2 + pad
    centers, grid_shape, vsize = vh.make_voxel_grid((lo, hi), args.res)
    print(f"[grid] res={args.res}^3 = {centers.shape[0]:,} voxels, voxel_size={vsize.round(3)}")

    # 4) carving
    occ = vh.carve(centers, views, cam_dist, K, (size, size))
    volume = vh.occupancy_to_volume(occ, grid_shape)
    n_occ = int(occ.sum())
    print(f"[carve] 남은 voxel {n_occ:,} / {centers.shape[0]:,} ({100*n_occ/centers.shape[0]:.1f}%)")
    if n_occ == 0:
        raise SystemExit("[ERROR] 빈 결과 — 카메라 포즈 정합 또는 임계값 점검 필요.")

    # 5) mesh
    recon = vh.volume_to_mesh(volume, vsize, origin=lo)
    print(f"[mesh] 복원 verts={len(recon.vertices):,} faces={len(recon.faces):,}")

    # 6) 정량 비교
    pitch = args.iou_pitch or float(max(extents) / 64.0)
    iou = met.voxel_iou(met.normalize_for_compare(recon),
                        met.normalize_for_compare(gt), pitch=pitch)
    cd = met.chamfer_distance(met.normalize_for_compare(recon),
                              met.normalize_for_compare(gt))
    result = {
        "part": os.path.basename(args.stl),
        "n_views": len(views),
        "voxel_res": args.res,
        "occupied_voxels": n_occ,
        "recon_faces": len(recon.faces),
        "iou": round(iou, 4),
        "chamfer": round(cd, 4),
        "iou_pitch": round(pitch, 4),
    }
    print("[지표] " + json.dumps(result, ensure_ascii=False))

    # 7) 저장
    if args.out:
        out = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        recon.export(out + "_hull.ply")
        with open(out + "_metrics.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[저장] {out}_hull.ply / {out}_metrics.json")


if __name__ == "__main__":
    main()
