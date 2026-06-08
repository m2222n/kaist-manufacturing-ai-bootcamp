#!/usr/bin/env python
"""
진단: GT mesh 의 정점을 한 뷰에 투영해서 그 뷰의 실루엣과 겹치는지 확인.
포즈가 맞으면 투영점이 실루엣 mask 안에 거의 다 들어가야 한다.
겹침 비율이 낮으면 카메라 포즈/투영 부호 문제.
"""
import os, sys, glob, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import camera as cam
from src import silhouette as sil

import trimesh

STL = os.path.expanduser("~/kaist_render/stl/02_sol_block_b.stl")
VDIR = os.path.expanduser("~/kaist_render/KAIST_dataset_v1/02_sol_block_b")

gt = trimesh.load(STL, force="mesh")
gt.apply_translation(-gt.bounding_box.centroid)
radius = float(np.linalg.norm(gt.bounding_box.extents) / 2.0)
cam_dist = radius / math.tan(math.radians(30)) * 1.05

paths = sorted(glob.glob(os.path.join(VDIR, "*.png")))
# 처음 몇 뷰만 검사
for p in paths[:4]:
    tilt, spin = cam.parse_angles(os.path.basename(p))
    mask = sil.load_silhouette(p)
    H, W = mask.shape
    K = cam.intrinsic_matrix(H)
    w2c = cam.view_extrinsic(tilt, spin, cam_dist)
    uv, z = cam.project(gt.vertices, w2c, K)
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    hit = np.zeros(len(u), bool)
    hit[inside] = mask[v[inside], u[inside]]
    print(f"y{int(tilt):03d}_z{int(spin):03d}: "
          f"투영 in-frame={inside.mean():.2f}, z>0={np.mean(z>0):.2f}, "
          f"실루엣 안 적중={hit.mean():.2f}, "
          f"u=[{u.min()},{u.max()}] v=[{v.min()},{v.max()}]")
