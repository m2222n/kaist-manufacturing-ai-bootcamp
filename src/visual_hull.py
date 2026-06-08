#!/usr/bin/env python
"""
Visual Hull (Shape-from-Silhouette) 코어.

원리: 각 뷰의 실루엣을 카메라 방향으로 역투영하면 실루엣 콘이 생긴다.
모든 뷰 콘의 교집합이 Visual Hull. voxel grid 의 각 voxel 중심을 모든 뷰에
투영해, "모든 뷰에서 전경 안"인 voxel 만 남기면(carving) Hull 이 된다.

한계(발표에서 정직하게 다룰 것):
- 내부 오목면/홈/공동(concavity)은 실루엣에 안 드러나 복원 불가 → convex 쪽으로 부풂.
- 점자프린터 부품은 구멍·슬롯이 많아 이 한계가 실제로 보일 수 있음.
"""
import numpy as np
from . import camera as cam


def make_voxel_grid(bounds, res):
    """
    bounds: (min_xyz, max_xyz) 각 (3,). res: 한 축 voxel 개수(int) 또는 (3,).
    반환: centers (M,3) 월드좌표, grid_shape (rx,ry,rz), voxel_size (3,).
    """
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    if np.isscalar(res):
        res = (int(res),) * 3
    rx, ry, rz = res
    # voxel 중심 좌표 (가장자리 반칸 안쪽)
    xs = np.linspace(lo[0], hi[0], rx)
    ys = np.linspace(lo[1], hi[1], ry)
    zs = np.linspace(lo[2], hi[2], rz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    voxel_size = (hi - lo) / (np.array(res) - 1)
    return centers, (rx, ry, rz), voxel_size


def carve(centers, views, cam_dist, K, mask_shape):
    """
    centers: (M,3) voxel 중심 (월드).
    views: list of (tilt_y_deg, spin_z_deg, mask) — mask 는 (H,W) bool.
    cam_dist: 렌더 카메라 거리 (부품별, render 와 동일 공식으로 계산).
    K: intrinsic (3,3).
    mask_shape: (H,W) — 투영 좌표 범위 체크용.
    반환: occupancy (M,) bool — 모든 뷰 실루엣 안에 있는 voxel만 True.
    """
    H, W = mask_shape
    occ = np.ones(centers.shape[0], dtype=bool)
    for tilt, spin, mask in views:
        w2c = cam.view_extrinsic(tilt, spin, cam_dist)
        uv, z = cam.project(centers, w2c, K)
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        # 화면 밖 또는 카메라 뒤 voxel 은 이 뷰에서 깎지 않음(보수적) → 안전하게 inside 만 검사
        fg = np.zeros(centers.shape[0], dtype=bool)
        fg[inside] = mask[v[inside], u[inside]]
        # 화면 밖으로 나간 voxel 은 실루엣 판정 불가 → 그 뷰에서는 통과시킴(깎지 않음).
        # 단, 카메라 뒤(z<=0)나 명백히 밖이면 깎는 게 맞으나, turntable 커버라
        # 대부분 화면 안에 들어오므로 보수적 정책 사용.
        keep_this_view = fg | (~inside)
        occ &= keep_this_view
    return occ


def occupancy_to_volume(occ, grid_shape):
    """occupancy (M,) → 3D 볼륨 (rx,ry,rz) bool."""
    return occ.reshape(grid_shape)


def volume_to_mesh(volume, voxel_size, origin):
    """
    binary 볼륨 → mesh (marching cubes). skimage 사용.
    반환: trimesh.Trimesh.
    """
    from skimage import measure
    import trimesh
    # marching_cubes 는 0.5 등치면. 볼륨이 비면 예외.
    if volume.sum() == 0:
        raise ValueError("빈 볼륨 — carving 결과가 비었습니다 (포즈/임계값 점검).")
    verts, faces, normals, _ = measure.marching_cubes(
        volume.astype(np.float32), level=0.5, spacing=tuple(voxel_size))
    verts = verts + np.asarray(origin)        # voxel index → 월드 좌표
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
