#!/usr/bin/env python
"""
28개 부품 전체에 Visual Hull 복원 → 부품별 정량 지표 일괄 산출.

발표 핵심 산출물: 부품별 IoU / Chamfer / gt_in_hull / hull_inflation 표.
이 분포가 "Visual Hull baseline 이 우리 부품에서 얼마나 통하는가"의 증거다.

사용:
  ./venv/bin/python scripts/run_all_visual_hull.py \
      --dataset ~/kaist_render/KAIST_dataset_v1 \
      --stl-dir ~/kaist_render/stl \
      --res 64 --out-dir outputs/all_res64

각 부품은 run_visual_hull 의 코어(carve→occupancy IoU)를 그대로 호출하므로
단일 실행과 수치가 일치한다.
"""
import os
import sys
import csv
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
    radius = float(np.linalg.norm(mesh.bounding_box.extents) / 2.0)
    if radius <= 0:
        radius = 1.0
    return radius / math.tan(math.radians(yfov_deg / 2.0)) * fit


def load_views(views_dir, bg_thresh):
    paths = sorted(glob.glob(os.path.join(views_dir, "*.png")))
    views, size = [], None
    for p in paths:
        tilt, spin = cam.parse_angles(os.path.basename(p))
        mask = sil.load_silhouette(p, bg_thresh=bg_thresh)
        if size is None:
            size = mask.shape[0]
        views.append((tilt, spin, mask))
    return views, size


def process_one(name, views_dir, stl_path, res, bg_thresh, out_dir):
    import trimesh
    gt = trimesh.load(stl_path, force="mesh")
    gt.apply_translation(-gt.bounding_box.centroid)
    extents = gt.bounding_box.extents
    cam_dist = compute_cam_dist(gt)

    views, size = load_views(views_dir, bg_thresh)
    if not views:
        return {"part": name, "error": "no_png"}
    K = cam.intrinsic_matrix(size)

    pad = 0.10 * extents
    lo, hi = -extents / 2 - pad, extents / 2 + pad
    centers, grid_shape, vsize = vh.make_voxel_grid((lo, hi), res)

    occ = vh.carve(centers, views, cam_dist, K, (size, size))
    volume = vh.occupancy_to_volume(occ, grid_shape)
    n_occ = int(occ.sum())
    if n_occ == 0:
        return {"part": name, "error": "empty_carve", "n_views": len(views)}

    occ_gt = met.gt_solid_voxels(gt, (lo, hi), res)
    iou = met.voxel_iou_occupancy(volume, occ_gt)
    cov = met.coverage_stats(volume, occ_gt)
    recon = vh.volume_to_mesh(volume, vsize, origin=lo)
    cd = met.chamfer_distance(recon, gt)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        recon.export(os.path.join(out_dir, f"{name}_hull.ply"))

    return {
        "part": name,
        "n_views": len(views),
        "voxel_res": res,
        "bbox": [round(float(e), 1) for e in extents],
        "occupied_voxels": n_occ,
        "gt_voxels": int(occ_gt.sum()),
        "iou": round(iou, 4),
        "chamfer": round(cd, 4),
        "gt_in_hull": round(cov["gt_in_hull"], 4),
        "hull_inflation": round(cov["hull_inflation"], 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="부품 폴더들이 든 데이터셋 루트")
    ap.add_argument("--stl-dir", required=True, help="원본 STL 폴더 (폴더명.stl 매칭)")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--bg-thresh", type=int, default=245)
    ap.add_argument("--out-dir", default=None, help="복원 mesh + 결과 표 저장 폴더")
    args = ap.parse_args()

    dataset = os.path.expanduser(args.dataset)
    stl_dir = os.path.expanduser(args.stl_dir)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else None

    part_dirs = sorted(d for d in glob.glob(os.path.join(dataset, "*")) if os.path.isdir(d))
    print(f"[배치] 부품 {len(part_dirs)}개, res={args.res}^3")

    rows = []
    for i, vd in enumerate(part_dirs, 1):
        name = os.path.basename(vd)
        stl = os.path.join(stl_dir, name + ".stl")
        if not os.path.exists(stl):
            print(f"  [{i:2}/{len(part_dirs)}] {name}  ⚠️ STL 없음 → 스킵")
            continue
        r = process_one(name, vd, stl, args.res, args.bg_thresh, out_dir)
        rows.append(r)
        if "error" in r:
            print(f"  [{i:2}/{len(part_dirs)}] {name}  ❌ {r['error']}")
        else:
            print(f"  [{i:2}/{len(part_dirs)}] {name:28} "
                  f"IoU={r['iou']:.3f}  CD={r['chamfer']:.2f}  "
                  f"gt_in_hull={r['gt_in_hull']:.3f}  infl={r['hull_inflation']:.2f}")

    ok = [r for r in rows if "error" not in r]
    if ok:
        ious = np.array([r["iou"] for r in ok])
        cds = np.array([r["chamfer"] for r in ok])
        ghs = np.array([r["gt_in_hull"] for r in ok])
        print("\n=== 요약 ({} 부품) ===".format(len(ok)))
        print(f"  IoU         mean={ious.mean():.3f}  median={np.median(ious):.3f}  "
              f"min={ious.min():.3f}  max={ious.max():.3f}")
        print(f"  Chamfer(mm) mean={cds.mean():.2f}   median={np.median(cds):.2f}")
        print(f"  gt_in_hull  mean={ghs.mean():.3f}   min={ghs.min():.3f}  "
              f"(1.0 에 가까울수록 GT 가 hull 안에 잘 들어감 = 정상)")

    if out_dir:
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        cols = ["part", "n_views", "bbox", "iou", "chamfer", "gt_in_hull",
                "hull_inflation", "occupied_voxels", "gt_voxels"]
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in ok:
                w.writerow([r.get(c, "") for c in cols])
        print(f"\n[저장] {out_dir}/summary.json + summary.csv + 부품별 _hull.ply")


if __name__ == "__main__":
    main()
