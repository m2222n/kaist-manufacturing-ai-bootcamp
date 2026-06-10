#!/usr/bin/env python
"""
STL 부품을 다각도로 렌더링하며 RGB + 실루엣 마스크 + Depth Map 을 정렬해 저장.

render_8views.py 의 자매 스크립트.
- 카메라 포즈/회전/intrinsic 규약은 render_8views.py / src/camera.py 와 100% 동일.
  (eye=[0,-cam_dist,0], yfov=60도, 부품을 rot_z 먼저 rot_y 나중, 파일명 y{tilt}_z{spin})
- 차이: pyrender 의 render() 가 원래 반환하던 depth 를 버리지 않고 저장한다.

배경 (6/9 KAIST 교수님 피드백):
  "CAD 에서 Depth Map 을 추출하라. 하나하나의 입력 이미지로 추론."
  Basler Blaze = ToF depth 카메라 → CAD 합성 depth 와 같은 modality 라
  RGB 보다 sim-to-real 도메인 갭이 작다. embedding/YOLO 입력 후보로 depth 가 유력.

⚠️ Depth 단위·정규화 (Blaze 실측과 정합할 때 반드시 일관시킬 것):
  - pyrender depth 는 카메라 z 거리(월드 단위 = STL 단위, 보통 **mm**)다. 배경(미적중)은 0.
  - near/far 를 부품 bounding sphere 기준으로 잡아 시각화 png 를 만든다.
  - 학습에 쓸 원본 거리값은 .npy (float32, mm) 로 보존한다. png 는 눈으로 보는 용도.
  - 단일 뷰 depth = 2.5D (앞면만). 합성 depth 는 너무 깨끗 → Blaze 노이즈/flying pixel
    augmentation 을 학습 시 별도로 섞어야 "외운 모델" 을 피한다 (메모리 경계 사항).

사용:
  PYOPENGL_PLATFORM=egl ./venv/bin/python scripts/render_depth.py \
      --in stl --out out_depth --views 8 --size 640 --bg 255

출력: out_depth/<부품명>/
  <부품명>_y{tilt}_z{spin}_rgb.png      RGB (render_8views 와 동일 룩)
  <부품명>_y{tilt}_z{spin}_mask.png     실루엣 (부품=255, 배경=0)
  <부품명>_y{tilt}_z{spin}_depth.npy    원본 거리 float32 (mm, 배경=0)
  <부품명>_y{tilt}_z{spin}_depth.png    시각화 (near=흰/far=검, 배경=검) — 확인용
  <부품명>_meta.json                    부품별 카메라/depth 메타 (정합·학습용)
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # 6000 헤드리스

import argparse
import json
import math
import glob
import numpy as np
import trimesh
import pyrender
from PIL import Image


def look_at(eye, target, up):
    """카메라 pose 행렬 (4x4) 생성. render_8views.look_at 과 동일."""
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


def depth_to_png(depth, znear, zfar):
    """거리 depth(float, 배경=0) → 8bit 시각화. 가까울수록 밝게, 배경은 검정."""
    vis = np.zeros(depth.shape, dtype=np.uint8)
    hit = depth > 0
    if hit.any():
        d = np.clip(depth[hit], znear, zfar)
        # near=1.0(흰), far=0.0(검) 으로 정규화
        norm = 1.0 - (d - znear) / max(zfar - znear, 1e-9)
        vis[hit] = (norm * 255.0).astype(np.uint8)
    return vis


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

    # 원점 중심으로 이동 (render_8views 와 동일)
    mesh.apply_translation(-mesh.bounding_box.centroid)
    radius = float(np.linalg.norm(mesh.bounding_box.extents) / 2.0)
    if radius <= 0:
        radius = 1.0

    cam_dist = radius / math.tan(math.radians(30)) * fit  # yfov=60도 가정
    # depth 클리핑 평면: 카메라 거리 ± 반지름 여유. 학습/시각화 정규화 기준.
    znear = max(cam_dist - radius * 1.5, radius * 0.01)
    zfar = cam_dist + radius * 1.5

    part_out = os.path.join(out_dir, name)
    os.makedirs(part_out, exist_ok=True)

    yfov = math.radians(60)
    bgf = bg / 255.0
    saved = 0
    d_global_min, d_global_max = math.inf, 0.0  # 실제 적중 depth 범위 기록용

    # 뷰 enumeration = render_8views.py 와 완전히 동일 (정각 회피 오프셋 포함)
    tilt_steps = n_views // 2 + 1          # n_views=8 → 5단계
    Y_OFFSET = 10.0
    Z_OFFSET = 10.0
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
                                   ambient_light=[0.10, 0.10, 0.10])
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.82, 0.84, 0.88, 1.0],
                metallicFactor=0.0, roughnessFactor=0.6)
            scene.add(pyrender.Mesh.from_trimesh(m, material=material, smooth=True))

            eye = [0.0, -cam_dist, 0.0]
            pose = look_at(eye, [0, 0, 0], [0, 0, 1])
            # znear/zfar 를 명시해 depth 가 클리핑 평면에 일관되도록.
            cam = pyrender.PerspectiveCamera(yfov=yfov, znear=znear, zfar=zfar)
            scene.add(cam, pose=pose)

            # 조명 = render_8views.py 와 동일 (RGB 룩 일치)
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
                # pyrender 는 (color, depth) 를 반환. depth = 카메라까지 거리(월드 단위), 배경=0.
                color, depth = r.render(scene)
            finally:
                r.delete()

            stem = f"{name}_y{int(y_ang):03d}_z{int(z_ang):03d}"
            # RGB
            Image.fromarray(color).save(os.path.join(part_out, f"{stem}_rgb.png"))
            # 실루엣 마스크 (depth 적중 영역 = 부품)
            mask = (depth > 0).astype(np.uint8) * 255
            Image.fromarray(mask).save(os.path.join(part_out, f"{stem}_mask.png"))
            # 원본 depth (학습용, float32 mm, 배경 0)
            np.save(os.path.join(part_out, f"{stem}_depth.npy"), depth.astype(np.float32))
            # depth 시각화 png (확인용)
            Image.fromarray(depth_to_png(depth, znear, zfar)).save(
                os.path.join(part_out, f"{stem}_depth.png"))

            hit = depth > 0
            if hit.any():
                d_global_min = min(d_global_min, float(depth[hit].min()))
                d_global_max = max(d_global_max, float(depth[hit].max()))
            saved += 1

    # 부품별 메타 — Blaze 실측 정합/학습 정규화 시 필요
    f_pix = size / (2.0 * math.tan(yfov / 2.0))
    meta = {
        "part": name,
        "n_views": saved,
        "image_size": size,
        "yfov_deg": 60.0,
        "intrinsic_K": [[f_pix, 0, size / 2.0],
                        [0, f_pix, size / 2.0],
                        [0, 0, 1.0]],
        "cam_dist": cam_dist,
        "bsphere_radius": radius,
        "znear": znear,
        "zfar": zfar,
        "depth_unit": "STL units (likely mm); background=0",
        "depth_hit_min": None if math.isinf(d_global_min) else d_global_min,
        "depth_hit_max": d_global_max,
        "note": "depth.npy = 원본 거리(학습용). depth.png = near밝음/far어둠 시각화. "
                "단일 뷰 = 2.5D. Blaze ToF 정합 시 노이즈 augmentation 필요.",
    }
    with open(os.path.join(part_out, f"{name}_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    print(f"  [OK]   {name}: {saved}뷰 × (rgb+mask+depth.npy+depth.png) + meta")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="stl")
    ap.add_argument("--out", dest="outdir", default="out_depth")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--bg", type=int, default=255, help="배경 밝기 0-255")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.stl")) +
                   glob.glob(os.path.join(args.indir, "*.STL")))
    if not files:
        print(f"STL 없음: {args.indir}/")
        return
    print(f"=== {len(files)}개 부품 × {args.views}각도 RGB+mask+depth 렌더 시작 ===")
    total = 0
    for p in files:
        total += render_part(p, args.outdir, args.views, args.size, args.bg)
    print(f"=== 완료: 총 {total}뷰 → {args.outdir}/ (뷰당 4파일 + 부품당 meta.json) ===")


if __name__ == "__main__":
    main()
