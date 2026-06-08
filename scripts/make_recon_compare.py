#!/usr/bin/env python
"""
③ Visual Hull 복원 vs 원본 STL 비교 이미지 (헤드리스 EGL 렌더).
잘되는 부품(11_sw_block)과 안되는 부품(17_mks_holder)을 같은 시점으로
[원본 | 복원] 나란히 → 발표에서 "Visual Hull 한계" 직관적으로 보여줌.
출력: docs/fig3_recon_compare.png
"""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
STL = os.path.expanduser("~/kaist_render/stl")
PLY = os.path.join(ROOT, "outputs/all_res64")

PAIRS = [("11_sw_block", "Good (IoU 0.81)"),
         ("17_mks_holder", "Hard (IoU 0.25)")]
W = 360


def render_mesh(mesh, color):
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounding_box.centroid)
    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.4, 0.4, 0.4])
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=color + [1.0],
                                             metallicFactor=0.1, roughnessFactor=0.7)
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=False))
    r = float(np.linalg.norm(mesh.bounding_box.extents) / 2.0) or 1.0
    d = r / np.tan(np.radians(20)) * 1.4
    # 3/4 시점 (형상 잘 보이게)
    cam = pyrender.PerspectiveCamera(yfov=np.radians(40))
    pose = np.eye(4)
    el, az = np.radians(25), np.radians(35)
    eye = np.array([d * np.cos(el) * np.sin(az), -d * np.cos(el) * np.cos(az), d * np.sin(el)])
    fwd = -eye / np.linalg.norm(eye)
    up = np.array([0, 0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    pose[:3, 0] = right; pose[:3, 1] = up2; pose[:3, 2] = -fwd; pose[:3, 3] = eye
    scene.add(cam, pose=pose)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    scene.add(light, pose=pose)
    rr = pyrender.OffscreenRenderer(W, W)
    flags = pyrender.RenderFlags.RGBA  # 알파 살려서 배경 흰색으로 합성
    color_img, _ = rr.render(scene, flags=flags)
    rr.delete()
    rgba = Image.fromarray(color_img, "RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def main():
    rows = len(PAIRS)
    canvas = Image.new("RGB", (W * 2 + 10, W * rows + 30 * rows), "white")
    dr = ImageDraw.Draw(canvas)
    for ri, (part, tag) in enumerate(PAIRS):
        gt = trimesh.load(os.path.join(STL, part + ".stl"), force="mesh")
        recon = trimesh.load(os.path.join(PLY, part + "_hull.ply"), force="mesh")
        img_gt = render_mesh(gt, [0.30, 0.45, 0.75])      # 파랑 = 원본
        img_rc = render_mesh(recon, [0.80, 0.45, 0.20])   # 주황 = 복원
        y = ri * (W + 30) + 25
        canvas.paste(img_gt, (0, y))
        canvas.paste(img_rc, (W + 10, y))
        dr.text((6, y - 20), f"{part}  [{tag}]", fill="black")
        dr.text((W // 2 - 30, y - 20 + W), "", fill="black")
    # 상단 컬럼 라벨
    dr.text((W // 2 - 30, 4), "Original (GT)", fill="#3050bf")
    dr.text((W + 10 + W // 2 - 30, 4), "Visual Hull recon", fill="#cc7020")
    out = os.path.join(DOCS, "fig3_recon_compare.png")
    canvas.save(out)
    print("[③] saved", out)


if __name__ == "__main__":
    main()
