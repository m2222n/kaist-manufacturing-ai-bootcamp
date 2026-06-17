#!/usr/bin/env python3
"""3D 좌표 추출 — 정규화까지 적용해 "실제로 쓸 수 있는 좌표"를 뽑는다.

probe_3d_coords.py가 찾은 두 문제를 해결:
  (1) 원점 offset (22/27이 메시 원점 형상 밖) -> COM을 원점으로 재배치(정규화)
  (2) 자세축 모호 -> 관성 주축으로 canonical frame 정의 + 고유값 내림차순 + 부호 규칙

그 다음 "3D 좌표"의 세 가지 해석별 산출물을 부품마다 생성해 본다:
  (A) 6DoF pose 기준: 정규화 원점(=COM) + 주축 회전(R) -> 부품별 canonical frame
  (B) 키포인트: canonical frame 기준 bbox 8코너 + 6 면중심 (학습 타깃 후보)
  (C) grasp 후보: 최대 단면 폭이 그리퍼 가능 범위인 축쌍 (단순 휴리스틱, 참고용)

목적 = 조교 보고용 "이런 좌표가 나옵니다" 실증 + JSON 산출.
"""
import glob, os, json
import numpy as np
import trimesh

STL_DIR = "/home/jtm/kaist_render/stl"
OUT_JSON = "/home/jtm/kaist_project/synth/coords_27parts.json"

def canonical_frame(m):
    """COM 원점 + 관성주축 정렬 = canonical frame 반환.
    R: world->canonical 회전(3x3), t: COM(원점이동). 부호 규칙으로 모호성 줄임."""
    com = np.array(m.center_mass)
    T = m.moment_inertia
    evals, evecs = np.linalg.eigh(T)            # evals 오름차순
    order = np.argsort(evals)[::-1]             # 내림차순(긴축 먼저)
    evals = evals[order]
    R = evecs[:, order].T                       # 행 = 주축
    # 부호 규칙: 각 주축이 +방향 정점을 더 많이 보도록 (모호성 일부 해소)
    verts_c = (np.array(m.vertices) - com) @ R.T
    for k in range(3):
        if np.sum(verts_c[:, k] > 0) < np.sum(verts_c[:, k] < 0):
            R[k, :] *= -1
            verts_c[:, k] *= -1
    # 우수성(det=+1) 보정
    if np.linalg.det(R) < 0:
        R[2, :] *= -1
        verts_c[:, 2] *= -1
    return com, R, evals, verts_c

def mat_to_quat(R):
    """3x3 -> quaternion wxyz (scipy 없이)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)

def process(path):
    name = os.path.basename(path)
    m = trimesh.load(path, force="mesh")
    com, R, evals, vc = canonical_frame(m)
    ext = vc.max(0) - vc.min(0)                 # canonical bbox 크기
    # (B) 키포인트: canonical bbox 8코너 + 6 면중심 (canonical 좌표계)
    lo, hi = vc.min(0), vc.max(0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                  for y in (lo[1], hi[1])
                                  for z in (lo[2], hi[2])])
    face_centers = np.array([
        [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]],
        [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, hi[2]],
        [(lo[0]+hi[0])/2, lo[1], (lo[2]+hi[2])/2],
        [(lo[0]+hi[0])/2, hi[1], (lo[2]+hi[2])/2],
        [lo[0], (lo[1]+hi[1])/2, (lo[2]+hi[2])/2],
        [hi[0], (lo[1]+hi[1])/2, (lo[2]+hi[2])/2],
    ])
    # (C) grasp 휴리스틱: 가장 짧은 두 축의 폭 = 집기 폭 후보
    widths = np.sort(ext)
    return {
        "name": name,
        "com_world_mm": [round(float(x), 3) for x in com],
        "principal_quat_wxyz": [round(float(x), 5) for x in mat_to_quat(R)],
        "canonical_bbox_mm": [round(float(x), 2) for x in ext],
        "n_keypoints": int(len(corners) + len(face_centers)),
        "grasp_min_width_mm": round(float(widths[0]), 2),
        "grasp_mid_width_mm": round(float(widths[1]), 2),
    }

def main():
    paths = sorted(glob.glob(os.path.join(STL_DIR, "*.stl")))
    rows = [process(p) for p in paths]
    print(f"=== 정규화 3D 좌표 산출 ({len(rows)}종) ===\n")
    hdr = f'{"부품":34} {"COM(world mm)":24} {"canon_bbox":18} {"grasp_w(min/mid)":16}'
    print(hdr); print("-"*len(hdr))
    for r in rows:
        com = ",".join(f"{v:.0f}" for v in r["com_world_mm"])
        bb = "x".join(f"{v:.0f}" for v in r["canonical_bbox_mm"])
        gw = f'{r["grasp_min_width_mm"]:.0f}/{r["grasp_mid_width_mm"]:.0f}'
        print(f'{r["name"]:34} {com:24} {bb:18} {gw:16}')
    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {OUT_JSON}")
    print(f"  - 각 부품: COM(위치 기준점) + 주축 쿼터니언(자세 기준) + canonical bbox + 키포인트 14개 + grasp 폭")
    # 그리퍼 가능 폭(가정 예: <40mm) 부품 수
    graspable = sum(1 for r in rows if r["grasp_min_width_mm"] < 40)
    print(f"  - grasp 최소폭<40mm(예시 그리퍼 가정): {graspable}/{len(rows)}종")

if __name__ == "__main__":
    main()
