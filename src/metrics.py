#!/usr/bin/env python
"""
복원 mesh vs 원본 STL 정량 비교.

⭐ 우리 강점: GT 3D(원본 STL)를 갖고 있어 복원 품질을 수치로 증명할 수 있다.
발표의 핵심 그래프(부품별 IoU / Chamfer)가 여기서 나온다.

- IoU: 두 mesh 를 같은 voxel grid 로 voxelize 후 교집합/합집합.
- Chamfer distance: 두 mesh 표면 샘플 점들 간 양방향 최근접 거리 평균.
"""
import numpy as np


def voxel_iou(mesh_a, mesh_b, pitch):
    """
    두 trimesh 를 동일 pitch 로 voxelize 후 IoU.
    pitch: voxel 한 변 길이(월드 단위). 작을수록 정밀하나 메모리 ↑.
    """
    # 공통 bounds 로 두 mesh 를 같은 격자에 올린다.
    lo = np.minimum(mesh_a.bounds[0], mesh_b.bounds[0])
    hi = np.maximum(mesh_a.bounds[1], mesh_b.bounds[1])

    def voxelize_to_grid(mesh):
        vg = mesh.voxelized(pitch=pitch).fill()
        # 점유 voxel 중심을 공통 격자 인덱스로 변환
        pts = vg.points                              # (K,3) 점유 voxel 중심
        idx = np.floor((pts - lo) / pitch).astype(int)
        return set(map(tuple, idx))

    a = voxelize_to_grid(mesh_a)
    b = voxelize_to_grid(mesh_b)
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def chamfer_distance(mesh_a, mesh_b, n_samples=20000):
    """
    양방향 Chamfer distance (월드 단위). 작을수록 가까움.
    표면을 균등 샘플 후 KDTree 최근접.
    """
    from scipy.spatial import cKDTree
    pa = mesh_a.sample(n_samples)
    pb = mesh_b.sample(n_samples)
    ta, tb = cKDTree(pa), cKDTree(pb)
    da, _ = tb.query(pa)        # a 점 → b 최근접
    db, _ = ta.query(pb)        # b 점 → a 최근접
    return float(da.mean() + db.mean())


def normalize_for_compare(mesh):
    """비교 전 정렬: 원점 중심 이동. (스케일은 둘 다 mm 단위 STL 이라 그대로.)"""
    m = mesh.copy()
    m.apply_translation(-m.bounding_box.centroid)
    return m
