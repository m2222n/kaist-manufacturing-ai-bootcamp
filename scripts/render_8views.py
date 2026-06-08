#!/usr/bin/env python
"""
STL 부품들을 8각도(45도 간격)로 회전 렌더링.
KAIST 빈피킹 데이터셋용 — CAD 부품별 다각도 합성 이미지.

사용:
  PYOPENGL_PLATFORM=egl ./venv/bin/python render_8views.py \
      --in stl --out out --views 8 --size 640 --bg 255

출력: out/<부품명>/<부품명>_<각도3자리>.png
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # 6000 헤드리스

import argparse
import math
import glob
import numpy as np
import trimesh
import pyrender
from PIL import Image


def look_at(eye, target, up):
    """카메라 pose 행렬 (4x4) 생성."""
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


def render_part(path, out_dir, n_views, size, bg, fit=1.05):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        mesh = trimesh.load(path, force="mesh")
    except Exception as e:
        print(f"  [SKIP] {name}: load 실패 ({e})")
        return 0
    if mesh.is_empty or len(mesh.vertices) == 0:
        print(f"  [SKIP] {name}: 빈 메시")
        return 0

    # 원점 중심으로 이동
    mesh.apply_translation(-mesh.bounding_box.centroid)
    radius = float(np.linalg.norm(mesh.bounding_box.extents) / 2.0)
    if radius <= 0:
        radius = 1.0

    cam_dist = radius / math.tan(math.radians(30)) * fit  # yfov=60도 가정
    part_out = os.path.join(out_dir, name)
    os.makedirs(part_out, exist_ok=True)

    yfov = math.radians(60)
    bgf = bg / 255.0
    saved = 0
    # 입체 부품 = 모든 각도 커버 (정각 회피):
    #   Y축 기울임 0~180° (반바퀴) × Z축 빙글 360° n_views
    #   균등 간격은 유지하되 시작 오프셋을 줘서 0/45/90 같은 "정각"을 피함.
    #   → 면이 카메라와 평행/수직이 안 돼 음영·구멍이 또렷이 드러남.
    #   전 방향 커버는 그대로라 빈피킹(임의 자세) 인식 학습에 안전.
    #   예) tilt_steps=5 → 10/52/94/136/178, 빙글 → 10/55/100/.../325
    tilt_steps = n_views // 2 + 1          # n_views=8 → 5단계
    Y_OFFSET = 10.0                        # 기울임 정각 회피 오프셋(도)
    Z_OFFSET = 10.0                        # 빙글 정각 회피 오프셋(도)
    for iy in range(tilt_steps):
        base_y = 180.0 / (tilt_steps - 1) * iy if tilt_steps > 1 else 0.0
        y_ang = base_y + Y_OFFSET
        rot_y = trimesh.transformations.rotation_matrix(math.radians(y_ang), [0, 1, 0])
        for iz in range(n_views):
            z_ang = 360.0 / n_views * iz + Z_OFFSET
            rot_z = trimesh.transformations.rotation_matrix(math.radians(z_ang), [0, 0, 1])
            m = mesh.copy()
            m.apply_transform(rot_z)   # 먼저 Z로 돌리고
            m.apply_transform(rot_y)   # 그 다음 Y로 기울임

            scene = pyrender.Scene(bg_color=[bgf, bgf, bgf, 1.0],
                                   ambient_light=[0.10, 0.10, 0.10])  # 면 밝기 살리되 구멍 안쪽은 어둡게
            # 밝은 회색 무광 재질 — 면 음영·구멍 깊이가 또렷이 드러나도록
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.82, 0.84, 0.88, 1.0],
                metallicFactor=0.0, roughnessFactor=0.6)
            scene.add(pyrender.Mesh.from_trimesh(m, material=material, smooth=True))

            # 카메라: 정면 시점 (FreeCAD에서 REAR 등 정투영 면을 똑바로 마주보듯).
            #   부품 면을 정면으로 봐서 구멍·슬롯이 또렷. 회전(Y/Z)으로 모든 면이 차례로 정면에 옴.
            eye = [0.0, -cam_dist, 0.0]
            pose = look_at(eye, [0, 0, 0], [0, 0, 1])
            cam = pyrender.PerspectiveCamera(yfov=yfov)
            scene.add(cam, pose=pose)

            # 3점 조명 (얕은 홈 강조 = grazing light):
            #   키라이트를 표면에 가깝게 낮춰(z 작게) 스치듯 비춤 → 자리파기·얕은 홈에도
            #   미세 그림자가 드리워져 드러남. 측면 비대칭 배치로 면 경사 명암차도 유지.
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.5),
                      pose=look_at([cam_dist * 1.5, -cam_dist * 0.8, cam_dist * 0.45],
                                   [0, 0, 0], [0, 0, 1]))
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.6),
                      pose=look_at([-cam_dist * 1.3, cam_dist * 0.3, cam_dist * 0.35],
                                   [0, 0, 0], [0, 0, 1]))
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.0),
                      pose=look_at([0, -cam_dist * 0.3, cam_dist * 1.6], [0, 0, 0], [0, 0, 1]))

            r = pyrender.OffscreenRenderer(size, size)
            try:
                color, _ = r.render(scene)
            finally:
                r.delete()
            fname = f"{name}_y{int(y_ang):03d}_z{int(z_ang):03d}.png"
            Image.fromarray(color).save(os.path.join(part_out, fname))
            saved += 1
    print(f"  [OK]   {name}: {saved}장")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="stl")
    ap.add_argument("--out", dest="outdir", default="out")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--bg", type=int, default=255, help="배경 밝기 0-255")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.stl")) +
                   glob.glob(os.path.join(args.indir, "*.STL")))
    if not files:
        print(f"STL 없음: {args.indir}/")
        return
    print(f"=== {len(files)}개 부품 × {args.views}각도 렌더 시작 ===")
    total = 0
    for p in files:
        total += render_part(p, args.outdir, args.views, args.size, args.bg)
    print(f"=== 완료: 총 {total}장 → {args.outdir}/ ===")


if __name__ == "__main__":
    main()
