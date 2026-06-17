#!/usr/bin/env python3
"""28부품 STL에서 3D 좌표값을 추출할 수 있는지 가능성 탐색.

6/11 KAIST 미팅 액션 ②: "28개 개별 부품을 3D 좌표값으로 뽑을 수 있는지 확인
(실루엣 X, 진짜 3D 좌표)". 조교 짐작 = CAD raw 코드로 가능할 것. 시작 전 조교 사전보고.

여기서 "3D 좌표"가 의미할 수 있는 세 가지를 전부 측정해 가능성을 본다:
  (A) 부품별 기준 좌표계 = centroid(질량중심) + COM(부피중심) + bbox center + 주축(PCA/관성)
      -> 6DoF pose의 "위치+자세 기준"을 STL에서 직접 정의 가능한지
  (B) 안정 자세(resting pose) 후보 = bbox 6면 중 어디로 놓이나 (X/Y 뒤집기 자세 enumeration용)
  (C) 키포인트 후보 = bbox 8코너 + 극점 (학습 타깃/grasp 후보로 쓸 수 있는지)

추출 신뢰도에 영향 주는 메시 품질(watertight, 원점 offset, 단위)도 같이 본다.
"""
import sys
import numpy as np
import trimesh

STL_DIR = "/home/jtm/kaist_render/stl"

import glob, os

def principal_axes(mesh):
    """관성 텐서 고유벡터 = 주축. 대칭/구형이면 불안정할 수 있어 고유값도 본다."""
    try:
        T = mesh.moment_inertia  # 3x3
        evals, evecs = np.linalg.eigh(T)
        return evals, evecs
    except Exception as e:
        return None, None

def probe_one(path):
    name = os.path.basename(path)
    m = trimesh.load(path, force="mesh")
    out = {"name": name}

    # --- 메시 품질 ---
    out["watertight"] = bool(m.is_watertight)
    out["n_faces"] = int(len(m.faces))
    ext = m.bounding_box.extents  # mm 가정 (STL 단위)
    out["bbox_mm"] = [round(float(x), 2) for x in ext]
    out["bbox_origin_offset_mm"] = round(float(np.linalg.norm(m.bounding_box.centroid)), 2)

    # --- (A) 기준 좌표계 후보 ---
    out["centroid_mm"] = [round(float(x), 2) for x in m.centroid]       # 면적가중 중심
    # COM(부피중심)은 watertight일 때만 신뢰
    try:
        com = m.center_mass
        out["com_mm"] = [round(float(x), 2) for x in com]
        out["volume_mm3"] = round(float(m.volume), 1)
    except Exception:
        out["com_mm"] = None
        out["volume_mm3"] = None
    out["bbox_center_mm"] = [round(float(x), 2) for x in m.bounding_box.centroid]

    # --- 주축 (자세 기준 축) ---
    evals, evecs = principal_axes(m)
    if evals is not None:
        out["inertia_evals"] = [round(float(x), 1) for x in evals]
        # 고유값이 거의 같으면(대칭/구형) 주축 방향이 ill-defined
        ev_sorted = np.sort(evals)
        ratio = float(ev_sorted[-1] / (ev_sorted[0] + 1e-9))
        out["axis_anisotropy"] = round(ratio, 2)  # 1에 가까우면 축 불안정
    else:
        out["inertia_evals"] = None
        out["axis_anisotropy"] = None

    # --- (B) 안정 자세 후보 개수 (convex hull face cluster 근사) ---
    try:
        # trimesh가 제공하는 정적 안정 자세 추정
        transforms, probs = trimesh.poses.compute_stable_poses(m)
        out["n_stable_poses"] = int(len(transforms))
        out["top_pose_prob"] = round(float(probs[0]), 3) if len(probs) else None
    except Exception as e:
        out["n_stable_poses"] = None
        out["top_pose_prob"] = None

    # --- (C) 키포인트 후보 = bbox 8 코너 ---
    out["n_bbox_corners"] = int(len(m.bounding_box.vertices))  # 항상 8
    return out

def main():
    paths = sorted(glob.glob(os.path.join(STL_DIR, "*.stl")))
    print(f"STL {len(paths)}개 탐색\n")
    rows = []
    fails = []
    for p in paths:
        try:
            rows.append(probe_one(p))
        except Exception as e:
            fails.append((os.path.basename(p), repr(e)))

    # 요약 표
    hdr = f'{"부품":34} {"wt":3} {"faces":6} {"bbox(mm)":22} {"origin_off":10} {"vol(mm3)":12} {"anis":6} {"stable":6}'
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        bbox = "x".join(f"{v:.0f}" for v in r["bbox_mm"])
        vol = f'{r["volume_mm3"]}' if r["volume_mm3"] is not None else "n/a"
        anis = f'{r["axis_anisotropy"]}' if r["axis_anisotropy"] is not None else "n/a"
        sp = f'{r["n_stable_poses"]}' if r["n_stable_poses"] is not None else "n/a"
        print(f'{r["name"]:34} {"Y" if r["watertight"] else "N":3} {r["n_faces"]:<6} '
              f'{bbox:22} {r["bbox_origin_offset_mm"]:<10} {vol:12} {anis:6} {sp:6}')

    # 통계
    n = len(rows)
    n_wt = sum(1 for r in rows if r["watertight"])
    n_com = sum(1 for r in rows if r["com_mm"] is not None)
    n_lowanis = sum(1 for r in rows if r["axis_anisotropy"] is not None and r["axis_anisotropy"] < 1.2)
    big_offset = [r["name"] for r in rows if r["bbox_origin_offset_mm"] > 50]
    print("\n=== 요약 ===")
    print(f"  전체 {n}개")
    print(f"  watertight(부피중심·안정자세 신뢰 가능): {n_wt}/{n}")
    print(f"  COM 추출 성공: {n_com}/{n}")
    print(f"  주축 거의 등방(축방향 불안정 우려, anis<1.2): {n_lowanis}/{n}")
    print(f"  원점 offset>50mm (좌표 기준 정규화 필요): {len(big_offset)}개 {big_offset}")
    if fails:
        print(f"\n  ❌ 로드 실패: {fails}")

if __name__ == "__main__":
    main()
