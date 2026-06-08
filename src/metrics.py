#!/usr/bin/env python
"""
복원 결과 vs 원본 STL 정량 비교.

⭐ 우리 강점: GT 3D(원본 STL)를 갖고 있어 복원 품질을 수치로 증명할 수 있다.
발표의 핵심 그래프(부품별 IoU / Chamfer)가 여기서 나온다.

⚠️ 설계 주의 (이전 버그 교훈):
  marching cubes 로 만든 mesh 는 watertight 가 아니라 .voxelized().fill() 로
  속을 못 채운다(껍데기만 잡혀 IoU 가 가짜로 낮게 나옴). 따라서 복원 쪽은
  **carving occupancy 볼륨(이미 solid)을 직접** 쓰고, GT 만 voxelize 한다.
  둘을 같은 좌표 격자에 올려 voxel 집합으로 IoU 를 잰다.

- voxel_iou_occupancy: carve occupancy(solid) vs GT solid voxel — 권장 경로.
- chamfer_distance: 두 표면 샘플 점 간 양방향 최근접 거리 (mesh 표면 비교).
"""
import numpy as np


def gt_solid_voxels(gt_mesh, bounds, res, chunk=20000):
    """
    GT mesh 를 carving 과 동일한 격자(bounds, res)에서 solid occupancy 로 voxelize.
    반환: occupancy (rx,ry,rz) bool.

    격자 중심점이 mesh 내부인지 contains() 로 판정 (rtree 필요).
    GT STL 은 watertight 이므로 contains 가 정확하다.

    ⚠️ contains() 는 점·면 수가 많으면 (예: 8000면 부품 × 26만 voxel) 중간 배열이
       수십 GB 로 폭발해 OOM 으로 죽는다 (6/8 17_mks_holder 58GB OOM 사고).
       점을 chunk 단위로 나눠 호출해 메모리를 상수로 묶는다. 결과는 동일.
    """
    from . import visual_hull as vh
    centers, grid_shape, _ = vh.make_voxel_grid(bounds, res)
    inside = np.empty(len(centers), dtype=bool)
    for i in range(0, len(centers), chunk):
        inside[i:i + chunk] = gt_mesh.contains(centers[i:i + chunk])
    return inside.reshape(grid_shape)


def voxel_iou_occupancy(occ_recon, occ_gt):
    """
    같은 격자의 두 occupancy 볼륨(bool, 동일 shape) IoU.
    occ_recon: carve 결과 볼륨. occ_gt: GT solid 볼륨.
    """
    assert occ_recon.shape == occ_gt.shape, "두 볼륨의 격자가 달라요 (같은 bounds/res 써야 함)"
    inter = np.logical_and(occ_recon, occ_gt).sum()
    union = np.logical_or(occ_recon, occ_gt).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def coverage_stats(occ_recon, occ_gt):
    """
    보조 지표: GT 가 hull 안에 얼마나 들어가나(recall), hull 이 얼마나 부풀었나.
    Visual Hull 은 GT ⊆ Hull 이 이상적 → gt_in_hull ≈ 1.0 이 정상.
    """
    gt_n = occ_gt.sum()
    recon_n = occ_recon.sum()
    if gt_n == 0:
        return {"gt_in_hull": 0.0, "hull_inflation": 0.0}
    gt_in_hull = np.logical_and(occ_recon, occ_gt).sum() / gt_n
    hull_inflation = recon_n / gt_n          # >1 이면 hull 이 GT 보다 큼(정상)
    return {"gt_in_hull": float(gt_in_hull), "hull_inflation": float(hull_inflation)}


def chamfer_distance(mesh_a, mesh_b, n_samples=20000):
    """
    양방향 Chamfer distance (월드 단위, mm). 작을수록 가까움.
    표면을 균등 샘플 후 KDTree 최근접. mesh 표면 형상 비교용.
    """
    from scipy.spatial import cKDTree
    pa = mesh_a.sample(n_samples)
    pb = mesh_b.sample(n_samples)
    ta, tb = cKDTree(pa), cKDTree(pb)
    da, _ = tb.query(pa)        # a 점 → b 최근접
    db, _ = ta.query(pb)        # b 점 → a 최근접
    return float(da.mean() + db.mean())
