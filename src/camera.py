#!/usr/bin/env python
"""
카메라 모델 + 뷰 포즈 추출.

render_8views.py 와 반드시 정합해야 한다. 렌더 스크립트는
**카메라를 고정**하고 **부품(물체)을 회전**시켜 다각도 이미지를 만든다.
Visual Hull 에서는 voxel(월드 좌표)을 각 뷰 이미지에 투영해야 하므로,
"물체를 R 만큼 회전"한 뷰는 "카메라를 R^{-1} 만큼 회전"한 것과 동치임을
이용해 world->camera extrinsic 을 복원한다.

핵심 정합 항목 (render_8views.py 기준):
- 카메라 위치 eye = [0, -cam_dist, 0], target=[0,0,0], up=[0,0,1]
- yfov = 60도, 정사각 이미지(size x size) → fx = fy = size / (2 tan(yfov/2))
- 뷰 = Z축 회전(빙글) 후 Y축 회전(기울임), 각도는 파일명 y{Y}_z{Z} 에 박혀 있음
- 부품에 적용한 순서: apply(rot_z) 먼저, 그 다음 apply(rot_y)
    => M_obj = rot_y @ rot_z  (열벡터 좌표 기준, 나중 적용이 왼쪽)
"""
import math
import re
import numpy as np

YFOV_DEG = 60.0          # render_8views.py PerspectiveCamera(yfov=60도)
DEFAULT_SIZE = 640       # 렌더 기본 --size (데이터셋은 800으로 뽑혔을 수 있음 → 파일에서 읽음)

# 파일명: <name>_y{tilt:03d}_z{spin:03d}.png
_FNAME_RE = re.compile(r"_y(\d{3})_z(\d{3})\.png$")


def rotation_matrix(angle_deg, axis):
    """오른손 좌표계 3x3 회전 행렬 (trimesh.rotation_matrix 의 3x3 부분과 동일)."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    return np.array([
        [c + x * x * (1 - c),     x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c),     y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def intrinsic_matrix(size, yfov_deg=YFOV_DEG):
    """정사각 perspective 카메라의 K (픽셀). 주점은 이미지 중심."""
    f = size / (2.0 * math.tan(math.radians(yfov_deg) / 2.0))
    cx = cy = size / 2.0
    return np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1.0],
    ])


def parse_angles(filename):
    """파일명에서 (tilt_y_deg, spin_z_deg) 추출. 없으면 ValueError."""
    m = _FNAME_RE.search(filename)
    if not m:
        raise ValueError(f"각도 파싱 실패: {filename}")
    return float(m.group(1)), float(m.group(2))


def camera_extrinsic_static():
    """
    렌더의 고정 카메라 (물체 회전 전 기준) 의 world->camera 변환 [R|t] (3x4).

    look_at(eye, target, up) 은 camera->world 포즈를 만든다 (pyrender 규약,
    카메라는 -Z 를 바라봄). world->camera 는 그 역.
    eye=[0,-d,0], target=0, up=[0,0,1].
    """
    eye = np.array([0.0, -1.0, 0.0])      # 방향만 필요 (거리는 아래서 곱함)
    # 아래 cam_dist 는 부품마다 다르므로 호출부에서 곱한 eye 를 넘기는 게 정확.
    # 여기서는 회전부(R_c2w)만 반환하고, t 는 view_extrinsic 에서 처리한다.
    raise NotImplementedError("view_extrinsic 를 사용하라")


def _c2w_from_lookat(eye, target, up):
    """render_8views.look_at 과 동일한 camera->world 4x4."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[:3, 0] = s
    m[:3, 1] = u
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m


def view_extrinsic(tilt_y_deg, spin_z_deg, cam_dist):
    """
    한 뷰의 world->camera extrinsic 4x4 를 반환.

    렌더는 부품을 M_obj = rot_y(tilt) @ rot_z(spin) 으로 돌린 뒤
    고정 카메라(eye=[0,-cam_dist,0])로 찍었다. 이는 부품을 원위치에 두고
    카메라를 M_obj^{-1} = rot_z(-spin) @ rot_y(-tilt) 로 옮긴 것과 동치.

    => 실제(월드 고정) 카메라 포즈(c2w):  M_obj^{-1} @ C2W_static
       world->camera = inv(그것)
    """
    c2w_static = _c2w_from_lookat([0, -cam_dist, 0], [0, 0, 0], [0, 0, 1])

    Ry = np.eye(4); Ry[:3, :3] = rotation_matrix(tilt_y_deg, [0, 1, 0])
    Rz = np.eye(4); Rz[:3, :3] = rotation_matrix(spin_z_deg, [0, 0, 1])
    M_obj = Ry @ Rz                       # 부품에 적용된 총 회전
    M_obj_inv = np.linalg.inv(M_obj)

    c2w = M_obj_inv @ c2w_static          # 월드(부품 원위치) 기준 카메라 포즈
    w2c = np.linalg.inv(c2w)
    return w2c                            # 4x4, [R|t; 0 0 0 1]


def project(points_world, w2c, K):
    """
    월드 3D 점들(N,3)을 한 뷰 이미지 픽셀(N,2)로 투영. (u, v) 와 depth 반환.
    pyrender/OpenGL 카메라는 -Z 를 바라보므로 카메라 좌표 z<0 이 앞쪽.
    """
    N = points_world.shape[0]
    hom = np.hstack([points_world, np.ones((N, 1))])      # (N,4)
    cam = (w2c @ hom.T).T[:, :3]                           # (N,3) 카메라 좌표
    z = -cam[:, 2]                                         # 앞쪽이 양수가 되도록
    # 핀홀 투영 (OpenGL: x_pix = fx * (X / -Z) + cx, y 는 위가 +Y 라 뒤집음)
    x = cam[:, 0] / (-cam[:, 2] + 1e-12)
    y = cam[:, 1] / (-cam[:, 2] + 1e-12)
    u = K[0, 0] * x + K[0, 2]
    v = -K[1, 1] * y + K[1, 2]            # 이미지 y축은 아래로 증가
    return np.stack([u, v], axis=1), z
